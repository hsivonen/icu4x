// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

// Various algorithms and constants in this file are adapted from
// ICU4C and, therefore, are subject to the ICU license as described
// in LICENSE.

#![cfg_attr(not(any(test, feature = "std")), no_std)]

//! `icu_collation` is one of the [`ICU4X`] components.
//!
//! This API provides necessary functionality for comparing strings according to language-dependent
//! conventions.
//!
//! # Design notes
//!
//! * The collation element design comes from ICU4C. Some parts of the ICU4C design, notably,
//!   `Tag::BuilderDataTag`, `Tag::LeadSurrogateTag`, `Tag::LatinExpansionTag`, `Tag::U0000Tag`,
//!   and `Tag::HangulTag` are unused.
//!   - `Tag::LatinExpansionTag` might be reallocated to search expansions for archaic jamo
//!     in the future.
//!   - `Tag::HangulTag` might be reallocated to compressed hanja expansions in the future.
//!     See https://github.com/unicode-org/icu4x/issues/1315
//! * The key design difference between ICU4C and ICU4X is that ICU4C puts the canonical
//!   closure in the data (larger data) to enable lookup directly by precomposed characters
//!   while ICU4X always omits the canonical closure and always normalizes to NFD on the fly.
//! * Compared to ICU4C, normalization cannot be turned off. There also isn't a separate
//!   "Fast Latin" mode.
//! * The normalization is fused into the collation element lookup algorithm to optimize the
//!   case where an input character decomposes into two BMP characters: a base letter and a
//!   diacritic.
//!   - To optimize away a trie  lookup when the combining diacritic doesn't contract,
//!     there is a linear lookup table for the combining diacritics block. Three languages
//!     tailor diacritics: Ewe, Lithuanian, and Vietnamese. Vietnamese and Ewe load and
//!     alternative table. The Lithuanian special cases are hard-coded activatable by
//!     a metadata bit.
//! * Unfortunately, contractions that contract starters don't fit this model nicely. Therefore,
//!   there's duplicated normalization code for normalizing the lookahead for contractions.
//!   This code can, in principle, do duplicative work, but it shouldn't be excessive with
//!   real-world inputs.
//! * As a result, in terms of code provenance, the algorithms come from ICU4C, except the
//!   normalization part of the code is novel to ICU4X, and the contraction code is custom
//!   to ICU4X despite being informed by ICU4C.
//! * The way input characters are iterated over and resulting collation elements are
//!   buffered is novel to ICU4X.
//! * ICU4C can iterate backwards but ICU4X cannot. ICU4X keeps a buffer of the two most
//!   recent characters for handling prefixes. As of CLDR 40, there were only two kinds
//!   of prefixes: a single starter and a starter followed by a kana voicing mark.
//! * ICU4C sorts unpaired surrogates in their lexical order. ICU4X operates on Unicode
//!   scalar values, so unpaired surrogates sort as REPLACEMENT CHARACTERs. Therefore,
//!   all unpaired surrogates are equal with each other.
//! * Skipping over a bit-identical prefix and then going back over "backward-unsafe"
//!   characters in currently unimplemented but isn't architecturally precluded. This
//!   is simply a TODO item for optimization.
//! * Hangul is handled specially:
//!   - Precomposed syllables are checked for as the first step of processing an
//!     incoming character.
//!   - Individual jamo are lookup up from a linear table instead of a trie. Unlike
//!     in ICU4C, this table covers the whole Unicode block whereas in ICU4C it covers
//!     only modern jamo for use in decomposing the precomposed syllables. The point
//!     is that search collations have a lot of duplicative (across multiple search)
//!     collations data for making archaic jamo searchable by modern jamo.
//!     Unfortunately, the shareable part isn't currently actually shareable, because
//!     the tailored CE32s refer to the expansions table in each collation. To make
//!     them truly shareable, the archaic jamo expansions need to become self-contained
//!     the way Latin mini expansions in ICU4C are self-contained.

pub mod error;
pub mod provider;

extern crate alloc;
use crate::provider::CollationDataV1Marker;
use crate::provider::CollationDiacriticsV1Marker;
use crate::provider::CollationJamoV1Marker;
use crate::provider::CollationMetadataV1Marker;
use crate::provider::CollationReorderingV1Marker;
use core::convert::TryFrom;
use icu_char16trie::char16trie::TrieResult;
use icu_locid::Locale;
use icu_normalizer::provider::CanonicalDecompositionDataV1;
use icu_normalizer::provider::CanonicalDecompositionDataV1Marker;
use icu_normalizer::Decomposition;
use icu_properties::provider::UnicodePropertyMapV1Marker;
use icu_properties::CanonicalCombiningClass;
use icu_provider::DataPayload;
use icu_provider::DataRequest;
use icu_provider::DynProvider;
use icu_provider::ResourceOptions;
use icu_provider::ResourceProvider;

use crate::error::CollatorError;
use crate::provider::CollationDataV1;

use core::char::{decode_utf16, DecodeUtf16Error, REPLACEMENT_CHARACTER};
use core::cmp::Ordering;
use icu_codepointtrie::CodePointTrie;

use smallvec::SmallVec;
use utf8_iter::Utf8CharsEx;
use zerovec::ule::AsULE;
use zerovec::ule::RawBytesULE;

// These constants originate from page 143 of Unicode 14.0
const HANGUL_S_BASE: u32 = 0xAC00;
const HANGUL_L_BASE: u32 = 0x1100;
const HANGUL_V_BASE: u32 = 0x1161;
const HANGUL_T_BASE: u32 = 0x11A7;
const HANGUL_T_COUNT: u32 = 28;
const HANGUL_N_COUNT: u32 = 588;
const HANGUL_S_COUNT: u32 = 11172;

const JAMO_COUNT: usize = 256; // 0x1200 - 0x1100

const COMBINING_DIACRITICS_BASE: usize = 0x0300;
const COMBINING_DIACRITICS_LIMIT: usize = 0x0370;
const COMBINING_DIACRITICS_COUNT: usize = COMBINING_DIACRITICS_LIMIT - COMBINING_DIACRITICS_BASE;

const CASE_MASK: u16 = 0xC000;
const TERTIARY_MASK: u16 = 0x3F3F; // ONLY_TERTIARY_MASK in ICU4C
const QUATERNARY_MASK: u16 = 0xC0;

const SPECIAL_CE32_LOW_BYTE: u8 = 0xC0;
const FALLBACK_CE32: CollationElement32 = CollationElement32(SPECIAL_CE32_LOW_BYTE as u32);
const LONG_PRIMARY_CE32_LOW_BYTE: u8 = 0xC1; // SPECIAL_CE32_LOW_BYTE | LONG_PRIMARY_TAG
const COMMON_SECONDARY_CE: u64 = 0x05000000;
const COMMON_TERTIARY_CE: u64 = 0x0500;
const COMMON_SEC_AND_TER_CE: u64 = COMMON_SECONDARY_CE | COMMON_TERTIARY_CE;

const UNASSIGNED_IMPLICIT_BYTE: u8 = 0xFE;

/// Set if there is no match for the single (no-suffix) character itself.
/// This is only possible if there is a prefix.
/// In this case, discontiguous contraction matching cannot add combining marks
/// starting from an empty suffix.
/// The default CE32 is used anyway if there is no suffix match.
// const CONTRACT_SINGLE_CP_NO_MATCH: u32 = 0x100;
/// Set if the first character of every contraction suffix has lccc!=0.
const CONTRACT_NEXT_CCC: u32 = 0x200;
/// Set if any contraction suffix ends with lccc!=0.
const CONTRACT_TRAILING_CCC: u32 = 0x400;
/// Set if at least one contraction suffix contains a starter
const CONTRACT_HAS_STARTER: u32 = 0x800;

// const NO_CE32: CollationElement32 = CollationElement32::const_default();
const NO_CE: CollationElement = CollationElement::const_default();
const NO_CE_PRIMARY: u32 = 1; // not a left-adjusted weight
                              // const NO_CE_NON_PRIMARY: NonPrimary = NonPrimary::const_default();
const NO_CE_SECONDARY: u16 = 0x0100;
const NO_CE_TERTIARY: u16 = 0x0100;

const MERGE_SEPARATOR_PRIMARY: u32 = 0x02000000; // U+FFFE

#[inline(always)]
fn in_inclusive_range(c: char, start: char, end: char) -> bool {
    u32::from(c).wrapping_sub(u32::from(start)) <= (u32::from(end) - u32::from(start))
}

/// Special-CE32 tags, from bits 3..0 of a special 32-bit CE.
/// Bits 31..8 are available for tag-specific data.
/// Bits  5..4: Reserved. May be used in the future to indicate lccc!=0 and tccc!=0.
#[derive(Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Tag {
    /// Fall back to the base collator.
    /// This is the tag value in SPECIAL_CE32_LOW_BYTE and FALLBACK_CE32.
    /// Bits 31..8: Unused, 0.
    FallbackTag = 0,
    /// Long-primary CE with COMMON_SEC_AND_TER_CE.
    /// Bits 31..8: Three-byte primary.
    LongPrimaryTag = 1,
    /// Long-secondary CE with zero primary.
    /// Bits 31..16: Secondary weight.
    /// Bits 15.. 8: Tertiary weight.
    LongSecondaryTag = 2,
    /// Unused.
    /// May be used in the future for single-byte secondary CEs (SHORT_SECONDARY_TAG),
    /// storing the secondary in bits 31..24, the ccc in bits 23..16,
    /// and the tertiary in bits 15..8.
    ReservedTag3 = 3,
    /// Latin mini expansions of two simple CEs [pp, 05, tt] [00, ss, 05].
    /// Bits 31..24: Single-byte primary weight pp of the first CE.
    /// Bits 23..16: Tertiary weight tt of the first CE.
    /// Bits 15.. 8: Secondary weight ss of the second CE.
    LatinExpansionTag = 4,
    /// Points to one or more simple/long-primary/long-secondary 32-bit CE32s.
    /// Bits 31..13: Index into uint32_t table.
    /// Bits 12.. 8: Length=1..31.
    Expansion32Tag = 5,
    /// Points to one or more 64-bit CEs.
    /// Bits 31..13: Index into CE table.
    /// Bits 12.. 8: Length=1..31.
    ExpansionTag = 6,
    /// Builder data, used only in the CollationDataBuilder, not in runtime data.
    ///
    /// If bit 8 is 0: Builder context, points to a list of context-sensitive mappings.
    /// Bits 31..13: Index to the builder's list of ConditionalCE32 for this character.
    /// Bits 12.. 9: Unused, 0.
    ///
    /// If bit 8 is 1 (IS_BUILDER_JAMO_CE32): Builder-only jamoCE32 value.
    /// The builder fetches the Jamo CE32 from the trie.
    /// Bits 31..13: Jamo code point.
    /// Bits 12.. 9: Unused, 0.
    BuilderDataTag = 7,
    /// Points to prefix trie.
    /// Bits 31..13: Index into prefix/contraction data.
    /// Bits 12.. 8: Unused, 0.
    PrefixTag = 8,
    /// Points to contraction data.
    /// Bits 31..13: Index into prefix/contraction data.
    /// Bits 12..11: Unused, 0.
    /// Bit      10: CONTRACT_TRAILING_CCC flag.
    /// Bit       9: CONTRACT_NEXT_CCC flag.
    /// Bit       8: CONTRACT_SINGLE_CP_NO_MATCH flag.
    ContractionTag = 9,
    /// Decimal digit.
    /// Bits 31..13: Index into uint32_t table for non-numeric-collation CE32.
    /// Bit      12: Unused, 0.
    /// Bits 11.. 8: Digit value 0..9.
    DigitTag = 10,
    /// Tag for U+0000, for moving the NUL-termination handling
    /// from the regular fastpath into specials-handling code.
    /// Bits 31..8: Unused, 0.
    U0000Tag = 11,
    /// Tag for a Hangul syllable.
    /// Bits 31..9: Unused, 0.
    /// Bit      8: HANGUL_NO_SPECIAL_JAMO flag.
    HangulTag = 12,
    /// Tag for a lead surrogate code unit.
    /// Optional optimization for UTF-16 string processing.
    /// Bits 31..10: Unused, 0.
    ///       9.. 8: =0: All associated supplementary code points are unassigned-implict.
    ///              =1: All associated supplementary code points fall back to the base data.
    ///              else: (Normally 2) Look up the data for the supplementary code point.
    LeadSurrogateTag = 13,
    /// Tag for CEs with primary weights in code point order.
    /// Bits 31..13: Index into CE table, for one data "CE".
    /// Bits 12.. 8: Unused, 0.
    ///
    /// This data "CE" has the following bit fields:
    /// Bits 63..32: Three-byte primary pppppp00.
    ///      31.. 8: Start/base code point of the in-order range.
    ///           7: Flag isCompressible primary.
    ///       6.. 0: Per-code point primary-weight increment.
    OffsetTag = 14,
    /// Implicit CE tag. Compute an unassigned-implicit CE.
    /// All bits are set (UNASSIGNED_CE32=0xffffffff).
    ImplicitTag = 15,
}

/// A compressed form of a collation element as stored in the collation
/// data.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct CollationElement32(u32);

impl CollationElement32 {
    #[inline(always)]
    pub const fn const_default() -> Self {
        CollationElement32(1)
    }

    #[inline(always)]
    pub fn new(bits: u32) -> Self {
        CollationElement32(bits)
    }

    #[inline(always)]
    pub fn new_from_ule(ule: RawBytesULE<4>) -> Self {
        CollationElement32(u32::from_unaligned(ule))
    }

    #[inline(always)]
    fn low_byte(&self) -> u8 {
        self.0 as u8
    }

    #[inline(always)]
    fn tag_checked(&self) -> Option<Tag> {
        let t = self.low_byte();
        if t < SPECIAL_CE32_LOW_BYTE {
            None
        } else {
            Some(self.tag())
        }
    }

    /// Returns the tag if this element is special.
    /// Non-specialness should first be checked by seeing if either
    /// `to_ce_simple_or_long_primary()` or `to_ce_self_contained()`
    /// returns non-`None`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if called on a non-special element.
    #[inline(always)]
    fn tag(&self) -> Tag {
        debug_assert!(self.low_byte() >= SPECIAL_CE32_LOW_BYTE);
        // By construction, the byte being transmuted to the enum is within
        // the value space of the enum, so the transmute cannot be UB.
        unsafe { core::mem::transmute(self.low_byte() & 0xF) }
    }

    /// Expands to 64 bits if the expansion is to a single 64-bit collation
    /// element and is not a long-secondary expansion.
    #[inline(always)]
    pub fn to_ce_simple_or_long_primary(&self) -> Option<CollationElement> {
        let t = self.low_byte();
        if t < SPECIAL_CE32_LOW_BYTE {
            let as64 = u64::from(self.0);
            Some(CollationElement::new(
                ((as64 & 0xFFFF0000) << 32) | ((as64 & 0xFF00) << 16) | (u64::from(t) << 8),
            ))
        } else if t == LONG_PRIMARY_CE32_LOW_BYTE {
            let as64 = u64::from(self.0);
            Some(CollationElement::new(
                ((as64 - u64::from(t)) << 32) | COMMON_SEC_AND_TER_CE,
            ))
        } else {
            None
        }
    }

    /// Expands to 64 bits if the expansion is to a single 64-bit collation
    /// element.
    #[inline(always)]
    pub fn to_ce_self_contained(&self) -> Option<CollationElement> {
        if let Some(ce) = self.to_ce_simple_or_long_primary() {
            return Some(ce);
        }
        if self.tag() == Tag::LongSecondaryTag {
            return Some(CollationElement::new(u64::from(self.0 & 0xffffff00)));
        } else {
            None
        }
    }

    /// Gets the length from this element.
    ///
    /// # Panics
    ///
    /// In debug builds if this element doesn't have a length.
    #[inline(always)]
    pub fn len(&self) -> usize {
        debug_assert!(self.tag() == Tag::Expansion32Tag || self.tag() == Tag::ExpansionTag);
        ((self.0 >> 8) & 31) as usize
    }

    /// Gets the index from this element.
    ///
    /// # Panics
    ///
    /// In debug builds if this element doesn't have an index.
    #[inline(always)]
    pub fn index(&self) -> usize {
        debug_assert!(
            self.tag() == Tag::Expansion32Tag
                || self.tag() == Tag::ExpansionTag
                || self.tag() == Tag::ContractionTag
                || self.tag() == Tag::DigitTag
                || self.tag() == Tag::PrefixTag
                || self.tag() == Tag::OffsetTag
        );
        (self.0 >> 13) as usize
    }

    #[inline(always)]
    pub fn digit(&self) -> u8 {
        debug_assert!(self.tag() == Tag::DigitTag);
        ((self.0 >> 8) & 0xF) as u8
    }

    #[inline(always)]
    pub fn every_suffix_starts_with_combining(&self) -> bool {
        debug_assert!(self.tag() == Tag::ContractionTag);
        (self.0 & CONTRACT_NEXT_CCC) != 0
    }
    #[inline(always)]
    pub fn at_least_one_suffix_contains_starter(&self) -> bool {
        debug_assert!(self.tag() == Tag::ContractionTag);
        (self.0 & CONTRACT_HAS_STARTER) != 0
    }
    #[inline(always)]
    pub fn at_least_one_suffix_ends_with_non_starter(&self) -> bool {
        debug_assert!(self.tag() == Tag::ContractionTag);
        (self.0 & CONTRACT_TRAILING_CCC) != 0
    }
}

impl Default for CollationElement32 {
    fn default() -> Self {
        CollationElement32(1) // NO_CE32
    }
}

/// A collation element is a 64-bit value.
///
/// The high 32 bits are the primary weight.
/// The high 16 bits of the low 32 bits are the secondary weight.
/// The high 2 bits of the low 16 bits are the case bits.
/// The high 2 bits of the low 8 bit are the quaternary weight.
/// The low 6 bits of the second to lowest 8 bits and the low 6
/// bits for the (bitwise discontiguous) tertiary weight.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CollationElement(u64);

impl CollationElement {
    #[inline(always)]
    pub fn new(bits: u64) -> Self {
        CollationElement(bits)
    }

    #[inline(always)]
    pub fn new_from_ule(ule: RawBytesULE<8>) -> Self {
        CollationElement(u64::from_unaligned(ule))
    }

    #[inline(always)]
    pub fn new_from_primary(primary: u32) -> Self {
        CollationElement((u64::from(primary) << 32) | COMMON_SEC_AND_TER_CE)
    }

    #[inline(always)]
    pub fn new_implicit_from_char(c: char) -> Self {
        // Collation::unassignedPrimaryFromCodePoint
        // Create a gap before U+0000. Use c-1 for [first unassigned].
        let mut c_with_offset = u32::from(c) + 1;
        // Fourth byte: 18 values, every 14th byte value (gap of 13).
        let mut primary: u32 = 2 + (c_with_offset % 18) * 14;
        c_with_offset /= 18;
        // Third byte: 254 values
        primary |= (2 + (c_with_offset % 254)) << 8;
        c_with_offset /= 254;
        // Second byte: 251 values 04..FE excluding the primary compression bytes.
        primary |= (4 + (c_with_offset % 251)) << 16;
        // One lead byte covers all code points (c < 0x1182B4 = 1*251*254*18).
        primary |= u32::from(UNASSIGNED_IMPLICIT_BYTE) << 24;
        CollationElement::new_from_primary(primary)
    }

    #[inline(always)]
    pub fn clone_with_non_primary_zeroed(&self) -> Self {
        CollationElement(self.0 & 0xFFFFFFFF00000000)
    }

    /// Get the primary weight
    #[inline(always)]
    pub fn primary(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Get the non-primary weight
    #[inline(always)]
    pub fn non_primary(&self) -> NonPrimary {
        NonPrimary::new(self.0 as u32)
    }

    /// Get the secondary weight
    #[inline(always)]
    pub fn secondary(&self) -> u16 {
        self.non_primary().secondary()
    }
    #[inline(always)]
    pub fn quaternary(&self) -> u32 {
        self.non_primary().quaternary()
    }
    #[inline(always)]
    pub fn tertiary_ignorable(&self) -> bool {
        self.non_primary().tertiary_ignorable()
    }
    #[inline(always)]
    pub fn either_half_zero(&self) -> bool {
        self.primary() == 0 || (self.0 as u32) == 0
    }

    #[inline(always)]
    pub const fn const_default() -> CollationElement {
        CollationElement(0x101000100)
    }
}

impl Default for CollationElement {
    #[inline(always)]
    fn default() -> Self {
        CollationElement(0x101000100) // NO_CE
    }
}

impl Default for &CollationElement {
    #[inline(always)]
    fn default() -> Self {
        &CollationElement(0x101000100) // NO_CE
    }
}

/// The purpose of grouping the non-primary bits
/// into a struct is to allow for a future optimization
/// the specializes code over whether storage for primary
/// weights is needed or not. (I.e. whether to specialize
/// on `CollationElement` or `NonPrimary`.)
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct NonPrimary(u32);

impl NonPrimary {
    pub fn new(bits: u32) -> Self {
        NonPrimary(bits)
    }
    /// Get the secondary weight
    #[inline(always)]
    pub fn secondary(&self) -> u16 {
        (self.0 >> 16) as u16
    }
    /// Get the case bits as the high two bits of a u16
    #[inline(always)]
    pub fn case(&self) -> u16 {
        (self.0 as u16) & CASE_MASK
    }
    /// Get the tertiary weight as u16 with the high
    /// two bits of each half zeroed.
    #[inline(always)]
    pub fn tertiary(&self) -> u16 {
        (self.0 as u16) & TERTIARY_MASK
    }
    #[inline(always)]
    pub fn tertiary_ignorable(&self) -> bool {
        (self.0 as u16) <= NO_CE_TERTIARY
    }
    /// Get the quaternary weight in the original
    /// storage bit positions with the other bits
    /// set to one.
    #[inline(always)]
    pub fn quaternary(&self) -> u32 {
        self.0 | !(QUATERNARY_MASK as u32)
    }
    /// Get any combination of tertiary, case, and quaternary
    /// by mask.
    #[inline(always)]
    pub fn tertiary_case_quarternary(&self, mask: u16) -> u16 {
        debug_assert!((mask & CASE_MASK) == CASE_MASK || (mask & CASE_MASK) == 0);
        debug_assert!((mask & TERTIARY_MASK) == TERTIARY_MASK || (mask & TERTIARY_MASK) == 0);
        debug_assert!((mask & QUATERNARY_MASK) == QUATERNARY_MASK || (mask & QUATERNARY_MASK) == 0);
        (self.0 as u16) & mask
    }

    #[inline(always)]
    pub fn case_quaternary(&self) -> u16 {
        (self.0 as u16) & (CASE_MASK | QUATERNARY_MASK)
    }

    #[inline(always)]
    pub const fn const_default() -> Self {
        NonPrimary(0x01000100) // Low 32 bits of NO_CE
    }
}

impl Default for NonPrimary {
    #[inline(always)]
    fn default() -> Self {
        NonPrimary(0x01000100) // Low 32 bits of NO_CE
    }
}

/// Pack a `char` and a `CanonicalCombiningClass` in
/// 32 bits. The latter is initialized to 0xFF upon
/// creation and can be set by calling `set_ccc`.
/// This type is intentionally non-`Copy` to get
/// compiler help in making sure that the class is
/// set on the instance on which it is intended to
/// be set and not on a temporary copy.
///
/// XXX check that 0xFF is actually reserved by the spec.
#[derive(Debug)]
struct CharacterAndClass(u32);

impl CharacterAndClass {
    pub fn new(c: char) -> Self {
        // Setting the placeholder combining class to 0xFF to
        // make it greater than zero when there is only one
        // combining character and we don't do the trie lookup.
        CharacterAndClass(u32::from(c) | (0xFF << 24))
    }
    pub fn character(&self) -> char {
        unsafe { char::from_u32_unchecked(self.0 & 0xFFFFFF) }
    }
    pub fn ccc(&self) -> CanonicalCombiningClass {
        CanonicalCombiningClass((self.0 >> 24) as u8)
    }
    // XXX need better naming here.
    pub fn set_ccc(&mut self, ccc: &CodePointTrie<CanonicalCombiningClass>) {
        debug_assert_eq!(self.0 >> 24, 0xFF, "This method has already been called!");
        let scalar = self.0 & 0xFFFFFF;
        self.0 = ((ccc.get(scalar).0 as u32) << 24) | scalar;
    }
}

// This trivial function exists as a borrow check helper.
#[inline(always)]
fn sort_slice_by_ccc<'data>(
    slice: &mut [char],
    ccc: &CodePointTrie<'data, CanonicalCombiningClass>,
) {
    slice.sort_by_key(|cc| ccc.get(u32::from(*cc)));
}

fn data_ce_to_primary(data_ce: u64, c: char) -> u32 {
    // Collation::getThreeBytePrimaryForOffsetData
    let p = (data_ce >> 32) as u32; // three-byte primary pppppp00
    let lower32 = data_ce as u32 as i32; // base code point b & step s: bbbbbbss (bit 7: isCompressible)
    let mut offset = ((u32::from(c) as i32) - (lower32 >> 8)) * (lower32 & 0x7F); // delta * increment
    let is_compressible = (lower32 & 0x80) != 0;
    // Collation::incThreeBytePrimaryByOffset
    offset += (((p >> 8) & 0xFF) as i32) - 2;
    let mut primary = (((offset % 254) + 2) as u32) << 8;
    offset /= 254;
    // Same with the second byte,
    // but reserve the PRIMARY_COMPRESSION_LOW_BYTE and high byte if necessary.
    if is_compressible {
        offset += (((p >> 16) & 0xFF) as i32) - 4;
        primary |= (((offset % 251) + 4) as u32) << 16;
        offset /= 251;
    } else {
        offset += (((p >> 16) & 0xFF) as i32) - 2;
        primary |= (((offset % 254) + 2) as u32) << 16;
        offset /= 254;
    }
    primary | ((p & 0xFF000000) + ((offset as u32) << 24))
}

/// Iterator that transforms an iterator over `char` into an iterator
/// over `CollationElement` with a tailoring.
/// Not a real Rust iterator: Instead of `None` uses `NO_CE` to indicate
/// end of iteration to optimize comparison.
struct CollationElements<'data, I>
where
    I: Iterator<Item = char>,
{
    iter: I,
    pending: SmallVec<[CollationElement; 6]>, // TODO Figure out good length
    // Two kinds of prefixes:
    // Prefixes that contain a single starter
    // Prefixes that contain a starter followed by either U+3099 or U+309A
    // Last-pushed is at index 0 and previously-pushed at index 1
    prefix: [char; 2],
    // Invariant: the last char of the suffix is a potentially-precomposed
    // starter unless iter has been exhausted.
    suffix: SmallVec<[char; 10]>, // TODO Figure out good length; longest contraction suffix is 7 characters long
    root: &'data CollationDataV1<'data>,
    tailoring: &'data CollationDataV1<'data>,
    // Note: in ICU4C the jamo table contains only modern jamo. Here, the jamo table contains the whole Unicode block.
    jamo: &'data [<u32 as AsULE>::ULE; JAMO_COUNT],
    diacritics: &'data [<u32 as AsULE>::ULE; COMBINING_DIACRITICS_COUNT],
    decompositions: &'data CanonicalDecompositionDataV1<'data>,
    ccc: &'data CodePointTrie<'data, CanonicalCombiningClass>,
    numeric: bool,
    lithuanian_dot_above: bool,
}

impl<'data, I> CollationElements<'data, I>
where
    I: Iterator<Item = char>,
{
    fn new(
        delegate: I,
        root: &'data CollationDataV1,
        tailoring: &'data CollationDataV1,
        jamo: &'data [<u32 as AsULE>::ULE; JAMO_COUNT],
        diacritics: &'data [<u32 as AsULE>::ULE; COMBINING_DIACRITICS_COUNT],
        decompositions: &'data CanonicalDecompositionDataV1,
        ccc: &'data CodePointTrie<'data, CanonicalCombiningClass>,
        numeric: bool,
        lithuanian_dot_above: bool,
    ) -> Self {
        // Invariant: Unless we're at the end, suffix contain at least
        // one character.
        let mut s = SmallVec::new();
        s.push('\u{FFFF}'); // Make sure the process always begins with a starter
        let mut ret = CollationElements::<I> {
            iter: delegate,
            pending: SmallVec::new(),
            prefix: ['\u{FFFF}'; 2],
            suffix: s,
            root: root,
            tailoring: tailoring,
            jamo: jamo,
            diacritics: diacritics,
            decompositions: decompositions,
            ccc: ccc,
            numeric: numeric,
            lithuanian_dot_above: lithuanian_dot_above,
        };
        let _ = ret.next(); // Remove the placeholder starter
        ret
    }

    fn next_internal(&mut self) -> Option<char> {
        if self.suffix.is_empty() {
            return None;
        }
        let ret = self.suffix.remove(0);
        if self.suffix.is_empty() {
            if let Some(c) = self.iter.next() {
                self.suffix.push(c);
            }
        }
        Some(ret)
    }

    fn maybe_gather_combining(&mut self) {
        if self.suffix.len() != 1 {
            return;
        }
        if !self
            .decompositions
            .decomposition_starts_with_non_starter
            .contains(self.suffix[0])
        {
            return;
        }
        while let Some(ch) = self.iter.next() {
            if self
                .decompositions
                .decomposition_starts_with_non_starter
                .contains(ch)
            {
                if !in_inclusive_range(ch, '\u{0340}', '\u{0F81}') {
                    self.suffix.push(ch);
                } else {
                    // The Tibetan special cases are starters that decompose into non-starters.
                    match ch {
                        '\u{0340}' => {
                            // COMBINING GRAVE TONE MARK
                            self.suffix.push('\u{0300}');
                        }
                        '\u{0341}' => {
                            // COMBINING ACUTE TONE MARK
                            self.suffix.push('\u{0301}');
                        }
                        '\u{0343}' => {
                            // COMBINING GREEK KORONIS
                            self.suffix.push('\u{0313}');
                        }
                        '\u{0344}' => {
                            // COMBINING GREEK DIALYTIKA TONOS
                            self.suffix.push('\u{0308}');
                            self.suffix.push('\u{0301}');
                        }
                        '\u{0F73}' => {
                            // TIBETAN VOWEL SIGN II
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F72}');
                        }
                        '\u{0F75}' => {
                            // TIBETAN VOWEL SIGN UU
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F74}');
                        }
                        '\u{0F81}' => {
                            // TIBETAN VOWEL SIGN REVERSED II
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F80}');
                        }
                        _ => {
                            self.suffix.push(ch);
                        }
                    };
                }
            } else {
                // Got a new starter
                self.suffix.push(ch);
                break;
            }
        }
    }

    // Decomposes `c`, pushes it to `self.suffix` (unless the character is
    // a Hangul syllable; Hangul isn't allowed to participate in contractions),
    // gathers the following combining characters from `self.iter` and the following starter.
    // Sorts the combining characters and leaves the starter at the end
    // unnormalized (maintaining the invariant of self.suffix). The
    // trailing starter doesn't get appended if `self.iter` is exhausted.
    fn push_decomposed_and_gather_combining(&mut self, c: char) {
        let mut search_start_combining = false;
        let old_len = self.suffix.len();
        // Not inserting early returns below to keep the same structure
        // as in the ce32 mapping code.

        // Hangul syllable check omitted, because it's fine not to decompose
        // Hangul syllables in lookahead, because Hangul isn't allowed to
        // participate in contractions, and the trie default is that a character
        // is its own decomposition.
        let decomposition = self.decompositions.trie.get(u32::from(c));
        if decomposition == 0 {
            // The character is its own decomposition (or Hangul syllable)
            self.suffix.push(c);
        } else {
            let high = (decomposition >> 16) as u16;
            let low = decomposition as u16;
            if high != 0 && low != 0 {
                // Decomposition into two BMP characters: starter and non-starter
                self.suffix
                    .push(core::char::from_u32(u32::from(high)).unwrap());
                self.suffix
                    .push(core::char::from_u32(u32::from(low)).unwrap());
            } else if high != 0 {
                // Decomposition into one BMP character
                self.suffix
                    .push(core::char::from_u32(u32::from(high)).unwrap());
            } else {
                // Complex decomposition
                let offset = usize::from(low & 0x7FF);
                let len = usize::from(low >> 13);
                if low & 0x1000 == 0 {
                    for &ule in
                        self.decompositions.scalars16.as_ule_slice()[offset..offset + len].iter()
                    {
                        self.suffix.push(
                            core::char::from_u32(u32::from(u16::from_unaligned(ule))).unwrap(),
                        );
                    }
                } else {
                    for &ule in
                        self.decompositions.scalars32.as_ule_slice()[offset..offset + len].iter()
                    {
                        self.suffix
                            .push(core::char::from_u32(u32::from_unaligned(ule)).unwrap());
                    }
                }
                if low & 0x800 != 0 {
                    search_start_combining = true;
                }
            }
        }
        let start_combining = if search_start_combining {
            // The decomposition contains starters. As of Unicode 14,
            // There are two possible patterns:
            // BMP: starter, starter, non-starter
            // Plane 1: starter, starter.
            // However, for forward compatility, support any combination
            // and search for the last starter.
            let mut i = self.suffix.len() - 1;
            while self
                .decompositions
                .decomposition_starts_with_non_starter
                .contains(self.suffix[i])
            {
                i -= 1;
            }
            i + 1
        } else {
            old_len + 1
        };
        let mut end_combining = start_combining;
        while let Some(ch) = self.iter.next() {
            if self
                .decompositions
                .decomposition_starts_with_non_starter
                .contains(ch)
            {
                if !in_inclusive_range(ch, '\u{0340}', '\u{0F81}') {
                    self.suffix.push(ch);
                } else {
                    // The Tibetan special cases are starters that decompose into non-starters.
                    match ch {
                        '\u{0340}' => {
                            // COMBINING GRAVE TONE MARK
                            self.suffix.push('\u{0300}');
                        }
                        '\u{0341}' => {
                            // COMBINING ACUTE TONE MARK
                            self.suffix.push('\u{0301}');
                        }
                        '\u{0343}' => {
                            // COMBINING GREEK KORONIS
                            self.suffix.push('\u{0313}');
                        }
                        '\u{0344}' => {
                            // COMBINING GREEK DIALYTIKA TONOS
                            self.suffix.push('\u{0308}');
                            self.suffix.push('\u{0301}');
                            end_combining += 1;
                        }
                        '\u{0F73}' => {
                            // XXX check if we can actually come here.
                            // TIBETAN VOWEL SIGN II
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F72}');
                            end_combining += 1;
                        }
                        '\u{0F75}' => {
                            // XXX check if we can actually come here.
                            // TIBETAN VOWEL SIGN UU
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F74}');
                            end_combining += 1;
                        }
                        '\u{0F81}' => {
                            // XXX check if we can actually come here.
                            // TIBETAN VOWEL SIGN REVERSED II
                            self.suffix.push('\u{0F71}');
                            self.suffix.push('\u{0F80}');
                            end_combining += 1;
                        }
                        _ => {
                            self.suffix.push(ch);
                        }
                    };
                }
                end_combining += 1;
            } else {
                // Got a new starter
                self.suffix.push(ch);
                break;
            }
        }
        // Perhaps there is a better borrow checker idiom than a function
        // call for indicating that `suffix` and `ccc` are disjoint and don't
        // overlap. However, this works.
        sort_slice_by_ccc(&mut self.suffix[start_combining..end_combining], self.ccc);
    }

    // Assumption: `pos` starts from zero and increases one by one.
    fn look_ahead(&mut self, pos: usize) -> Option<char> {
        if pos + 1 == self.suffix.len() {
            let c = self.suffix.remove(pos);
            self.push_decomposed_and_gather_combining(c);
            Some(self.suffix[pos])
        } else if pos == self.suffix.len() {
            if let Some(c) = self.iter.next() {
                self.push_decomposed_and_gather_combining(c);
                Some(self.suffix[pos])
            } else {
                None
            }
        } else {
            Some(self.suffix[pos])
        }
    }

    fn is_next_decomposition_starts_with_starter(&self) -> bool {
        if self.suffix.is_empty() {
            return true;
        }
        !self
            .decompositions
            .decomposition_starts_with_non_starter
            .contains(self.suffix[0])
    }

    fn prepend_and_sort_non_starter_prefix_of_suffix(&mut self, c: char) {
        // Add one for the insertion afterwards.
        let end = 1 + {
            let mut iter = self.suffix.iter().enumerate();
            loop {
                if let Some((i, &ch)) = iter.next() {
                    if !self
                        .decompositions
                        .decomposition_starts_with_non_starter
                        .contains(ch)
                    {
                        break i;
                    }
                } else {
                    break self.suffix.len();
                }
            }
        };
        self.suffix.insert(0, c);
        let start = if self
            .decompositions
            .decomposition_starts_with_non_starter
            .contains(c)
        {
            0
        } else {
            1
        };
        sort_slice_by_ccc(&mut self.suffix[start..end], self.ccc);
    }

    fn prefix_push(&mut self, c: char) {
        self.prefix[1] = self.prefix[0];
        self.prefix[0] = c;
    }

    /// Micro optimization for doing a simpler write when
    /// we know the most recent character was a non-starter
    /// that is not a kana voicing mark.
    fn mark_prefix_unmatchable(&mut self) {
        self.prefix[0] = '\u{FFFF}';
    }

    pub fn next(&mut self) -> CollationElement {
        if !self.pending.is_empty() {
            return self.pending.remove(0);
        }
        if let Some(mut c) = self.next_internal() {
            let mut next_is_known_to_decompose_to_non_starter = false; // micro optimization to avoid checking trie twice
            let mut ce32;
            let mut data: &CollationDataV1 = self.tailoring;
            let mut combining_characters: SmallVec<[CharacterAndClass; 7]> = SmallVec::new(); // XXX figure out proper size

            // Betting that fusing the NFD algorithm into this one at the
            // expense of the repetitiveness below, the common cases become
            // fast in a way that offsets the lack of the canonical closure.
            // The wall of code before the "Slow path" is an attempt to
            // optimize based on that bet.
            // TODO: ASCII fast path here if ASCII not tailored with
            // starters.
            let hangul_offset = u32::from(c).wrapping_sub(HANGUL_S_BASE); // SIndex in the spec
            if hangul_offset >= HANGUL_S_COUNT {
                let decomposition = self.decompositions.trie.get(u32::from(c));
                if decomposition == 0 {
                    // The character is its own decomposition
                    let jamo_index = (c as usize).wrapping_sub(HANGUL_L_BASE as usize);
                    if jamo_index >= self.jamo.len() {
                        ce32 = data.ce32_for_char(c);
                        if ce32 == FALLBACK_CE32 {
                            data = self.root;
                            ce32 = data.ce32_for_char(c);
                        }
                    } else {
                        // The purpose of reading the CE32 from the jamo table instead
                        // of the trie even in this case is to make it unnecessary
                        // for all search collation tries to carry a copy of the Hangul
                        // part of the search root. Instead, all non-Korean tailorings
                        // can use a shared copy of the non-Korean search jamo table.
                        //
                        // XXX This isn't actually true with the current jamo search
                        // expansions!

                        // We need to set data to root, because archaic jamo refer to
                        // the root.
                        data = self.root;
                        ce32 = CollationElement32::new_from_ule(self.jamo[jamo_index]);
                    }
                    if self.is_next_decomposition_starts_with_starter() {
                        if let Some(ce) = ce32.to_ce_simple_or_long_primary() {
                            self.prefix_push(c);
                            return ce;
                        } else if ce32.tag() == Tag::ContractionTag
                            && ce32.every_suffix_starts_with_combining()
                        {
                            // Avoid falling onto the slow path e.g. that letters that
                            // may contract with a diacritic when we know that it won't
                            // contract with the next character.
                            let default = data.get_default(ce32.index());
                            if let Some(ce) = default.to_ce_simple_or_long_primary() {
                                self.prefix_push(c);
                                return ce;
                            }
                        }
                    } else {
                        next_is_known_to_decompose_to_non_starter = true;
                    }
                } else {
                    let high = (decomposition >> 16) as u16;
                    let low = decomposition as u16;
                    if high != 0 && low != 0 {
                        // Decomposition into two BMP characters: starter and non-starter
                        c = core::char::from_u32(u32::from(high)).unwrap();
                        ce32 = data.ce32_for_char(c);
                        if ce32 == FALLBACK_CE32 {
                            data = self.root;
                            ce32 = data.ce32_for_char(c);
                        }
                        let combining = core::char::from_u32(u32::from(low)).unwrap();
                        if self.is_next_decomposition_starts_with_starter() {
                            let diacritic_index =
                                (low as usize).wrapping_sub(COMBINING_DIACRITICS_BASE);
                            if diacritic_index < self.diacritics.len() {
                                debug_assert!(low != 0x0344, "Should never have COMBINING GREEK DIALYTIKA TONOS here, since it should have decomposed further.");
                                if let Some(ce) = ce32.to_ce_simple_or_long_primary() {
                                    // Inner unwrap: already checked len()
                                    // Outer unwrap: expectation of data integrity.
                                    let ce_for_combining = CollationElement32::new_from_ule(
                                        self.diacritics[diacritic_index],
                                    )
                                    .to_ce_self_contained()
                                    .unwrap();
                                    self.pending.push(ce_for_combining);
                                    self.mark_prefix_unmatchable();
                                    return ce;
                                }
                                if ce32.tag() == Tag::ContractionTag
                                    && ce32.every_suffix_starts_with_combining()
                                {
                                    let (default, mut trie) =
                                        data.get_default_and_trie(ce32.index());
                                    match trie.next(combining) {
                                        TrieResult::NoMatch | TrieResult::NoValue => {
                                            if let Some(ce) = default.to_ce_simple_or_long_primary()
                                            {
                                                // Inner unwrap: already checked len()
                                                // Outer unwrap: expectation of data integrity.
                                                let ce_for_combining =
                                                    CollationElement32::new_from_ule(
                                                        self.diacritics[diacritic_index],
                                                    )
                                                    .to_ce_self_contained()
                                                    .unwrap();
                                                self.pending.push(ce_for_combining);
                                                self.mark_prefix_unmatchable();
                                                return ce;
                                            }
                                        }
                                        TrieResult::Intermediate(trie_ce32)
                                        | TrieResult::FinalValue(trie_ce32) => {
                                            // Assuming that we don't have longer matches with
                                            // a starter at this point. XXX Is this true?
                                            if let Some(ce) =
                                                CollationElement32::new(trie_ce32 as u32)
                                                    .to_ce_simple_or_long_primary()
                                            {
                                                self.mark_prefix_unmatchable();
                                                return ce;
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            next_is_known_to_decompose_to_non_starter = true;
                        }
                        combining_characters.push(CharacterAndClass::new(combining));
                    } else if high != 0 {
                        // Decomposition into one BMP character
                        c = core::char::from_u32(u32::from(high)).unwrap();
                        ce32 = data.ce32_for_char(c);
                        if ce32 == FALLBACK_CE32 {
                            data = self.root;
                            ce32 = data.ce32_for_char(c);
                        }
                        if self.is_next_decomposition_starts_with_starter() {
                            if let Some(ce) = ce32.to_ce_simple_or_long_primary() {
                                self.prefix_push(c);
                                return ce;
                            }
                        } else {
                            next_is_known_to_decompose_to_non_starter = true;
                        }
                    } else {
                        // Complex decomposition
                        let offset = usize::from(low & 0x7FF);
                        let len = usize::from(low >> 13);
                        if low & 0x1000 == 0 {
                            let (&first, tail) = &self.decompositions.scalars16.as_ule_slice()
                                [offset..offset + len]
                                .split_first()
                                .unwrap();
                            c = core::char::from_u32(u32::from(u16::from_unaligned(first)))
                                .unwrap();
                            if low & 0x800 == 0 {
                                for &ule in tail.iter() {
                                    combining_characters.push(CharacterAndClass::new(
                                        core::char::from_u32(u32::from(u16::from_unaligned(ule)))
                                            .unwrap(),
                                    ));
                                }
                            } else {
                                next_is_known_to_decompose_to_non_starter = false;
                                let mut it = tail.iter();
                                while let Some(&ule) = it.next() {
                                    let ch =
                                        core::char::from_u32(u32::from(u16::from_unaligned(ule)))
                                            .unwrap();
                                    if self
                                        .decompositions
                                        .decomposition_starts_with_non_starter
                                        .contains(ch)
                                    {
                                        // As of Unicode 14, this branch is never taken.
                                        // It exist for forward compatibility.
                                        combining_characters.push(CharacterAndClass::new(ch));
                                        continue;
                                    }

                                    // At this point, we might have a single newly-read
                                    // combining character in self.suffix. In that case, we
                                    // need to buffer up the upcoming combining characters, too,
                                    // in order to make `prepend_and_sort_non_starter_prefix_of_suffix`
                                    // sort the right characters.
                                    self.maybe_gather_combining();

                                    while let Some(&ule) = it.next_back() {
                                        self.prepend_and_sort_non_starter_prefix_of_suffix(
                                            core::char::from_u32(u32::from(u16::from_unaligned(
                                                ule,
                                            )))
                                            .unwrap(),
                                        );
                                    }
                                    self.prepend_and_sort_non_starter_prefix_of_suffix(ch);
                                    break;
                                }
                            }
                        } else {
                            let (&first, tail) = &self.decompositions.scalars32.as_ule_slice()
                                [offset..offset + len]
                                .split_first()
                                .unwrap();
                            c = core::char::from_u32(u32::from_unaligned(first)).unwrap();
                            if low & 0x800 == 0 {
                                for &ule in tail.iter() {
                                    combining_characters.push(CharacterAndClass::new(
                                        core::char::from_u32(u32::from_unaligned(ule)).unwrap(),
                                    ));
                                }
                            } else {
                                next_is_known_to_decompose_to_non_starter = false;
                                let mut it = tail.iter();
                                while let Some(&ule) = it.next() {
                                    let ch =
                                        core::char::from_u32(u32::from_unaligned(ule)).unwrap();
                                    if self
                                        .decompositions
                                        .decomposition_starts_with_non_starter
                                        .contains(ch)
                                    {
                                        // As of Unicode 14, this branch is never taken.
                                        // It exist for forward compatibility.
                                        combining_characters.push(CharacterAndClass::new(ch));
                                        continue;
                                    }
                                    // At this point, we might have a single newly-read
                                    // combining character in self.suffix. In that case, we
                                    // need to buffer up the upcoming combining characters, too,
                                    // in order to make `prepend_and_sort_non_starter_prefix_of_suffix`
                                    // sort the right characters.
                                    self.maybe_gather_combining();

                                    while let Some(&ule) = it.next_back() {
                                        self.prepend_and_sort_non_starter_prefix_of_suffix(
                                            core::char::from_u32(u32::from_unaligned(ule)).unwrap(),
                                        );
                                    }
                                    self.prepend_and_sort_non_starter_prefix_of_suffix(ch);
                                    break;
                                }
                            }
                        }
                        ce32 = data.ce32_for_char(c);
                        if ce32 == FALLBACK_CE32 {
                            data = self.root;
                            ce32 = data.ce32_for_char(c);
                        }
                    }
                }
            } else {
                // Hangul syllable
                // The math here comes from page 144 of Unicode 14.0
                let l = hangul_offset / HANGUL_N_COUNT;
                let v = (hangul_offset % HANGUL_N_COUNT) / HANGUL_T_COUNT;
                let t = hangul_offset % HANGUL_T_COUNT;

                // No prefix matches on Hangul
                self.mark_prefix_unmatchable();
                if self.is_next_decomposition_starts_with_starter() {
                    // XXX Figure out if non-self-contained jamo CE32s exist in
                    // CLDR for modern jamo.
                    self.pending.push(
                        CollationElement32::new_from_ule(
                            self.jamo[(HANGUL_V_BASE - HANGUL_L_BASE + v) as usize],
                        )
                        .to_ce_self_contained()
                        .unwrap(),
                    );
                    if t != 0 {
                        self.pending.push(
                            CollationElement32::new_from_ule(
                                self.jamo[(HANGUL_T_BASE - HANGUL_L_BASE + t) as usize],
                            )
                            .to_ce_self_contained()
                            .unwrap(),
                        );
                    }
                    return CollationElement32::new_from_ule(self.jamo[l as usize])
                        .to_ce_self_contained()
                        .unwrap();
                }

                // Uphold the invariant that the upcoming character is a starter or end of stream.
                // Not gathering combining characters into self.suffix is OK, because jamo
                // is guaranteed not to start a contraction.
                if t != 0 {
                    self.pending.push(
                        CollationElement32::new_from_ule(
                            self.jamo[(HANGUL_V_BASE - HANGUL_L_BASE + v) as usize],
                        )
                        .to_ce_self_contained()
                        .unwrap(),
                    );
                    self.suffix
                        .insert(0, core::char::from_u32(HANGUL_T_BASE + t).unwrap());
                } else {
                    self.suffix
                        .insert(0, core::char::from_u32(HANGUL_V_BASE + v).unwrap());
                }

                return CollationElement32::new_from_ule(self.jamo[l as usize])
                    .to_ce_self_contained()
                    .unwrap();
            }
            // Slow path
            while next_is_known_to_decompose_to_non_starter
                || !self.is_next_decomposition_starts_with_starter()
            {
                next_is_known_to_decompose_to_non_starter = false;
                let combining = self.next_internal().unwrap();
                if !in_inclusive_range(combining, '\u{0340}', '\u{0F81}') {
                    combining_characters.push(CharacterAndClass::new(combining));
                } else {
                    // The Tibetan special cases are starters that decompose into non-starters.
                    //
                    // Logically the canonical combining class of each special case is known
                    // at compile time, but all characters in the buffer are treated the same
                    // when looking up the canonical combining class to avoid a per-character
                    // branch that would only benefit these rare special cases.
                    match combining {
                        '\u{0340}' => {
                            // COMBINING GRAVE TONE MARK
                            combining_characters.push(CharacterAndClass::new('\u{0300}'));
                        }
                        '\u{0341}' => {
                            // COMBINING ACUTE TONE MARK
                            combining_characters.push(CharacterAndClass::new('\u{0301}'));
                        }
                        '\u{0343}' => {
                            // COMBINING GREEK KORONIS
                            combining_characters.push(CharacterAndClass::new('\u{0313}'));
                        }
                        '\u{0344}' => {
                            // COMBINING GREEK DIALYTIKA TONOS
                            combining_characters.push(CharacterAndClass::new('\u{0308}'));
                            combining_characters.push(CharacterAndClass::new('\u{0301}'));
                        }
                        '\u{0F73}' => {
                            // TIBETAN VOWEL SIGN II
                            combining_characters.push(CharacterAndClass::new('\u{0F71}'));
                            combining_characters.push(CharacterAndClass::new('\u{0F72}'));
                        }
                        '\u{0F75}' => {
                            // TIBETAN VOWEL SIGN UU
                            combining_characters.push(CharacterAndClass::new('\u{0F71}'));
                            combining_characters.push(CharacterAndClass::new('\u{0F74}'));
                        }
                        '\u{0F81}' => {
                            // TIBETAN VOWEL SIGN REVERSED II
                            combining_characters.push(CharacterAndClass::new('\u{0F71}'));
                            combining_characters.push(CharacterAndClass::new('\u{0F80}'));
                        }
                        _ => {
                            combining_characters.push(CharacterAndClass::new(combining));
                        }
                    };
                }
            }
            if combining_characters.len() > 1 {
                // This optimizes away the class lookup when len() == 1.
                // Unclear if this micro optimization is worthwhile.
                // In any case, we store the CanonicalCombiningClass in order to
                // avoid having to look it up again when deciding whether to proceed
                // with a discontiguous match. As a side effect, it also means that
                // duplicate lookups aren't needed if the sort below happens to compare
                // an item more than once.
                combining_characters
                    .iter_mut()
                    .for_each(|cc| cc.set_ccc(self.ccc));
                combining_characters.sort_by_key(|cc| cc.ccc());
            }
            // Now:
            // c is the starter character
            // ce32 is the CollationElement32 for the starter
            // combining_characters contains all the combining characters before
            // the next starter sorted by combining class.
            let mut looked_ahead = 0;
            let mut drain_from_suffix = 0;
            'outer: loop {
                'ce32loop: loop {
                    if let Some(ce) = ce32.to_ce_self_contained() {
                        self.pending.push(ce);
                        break 'ce32loop;
                    } else {
                        match ce32.tag() {
                            Tag::Expansion32Tag => {
                                let ce32s = data.get_ce32s(ce32.index(), ce32.len());
                                for &ce32_ule in ce32s {
                                    self.pending.push(
                                        CollationElement32::new_from_ule(ce32_ule)
                                            .to_ce_self_contained()
                                            .unwrap(),
                                    );
                                }
                                break 'ce32loop;
                            }
                            Tag::ExpansionTag => {
                                let ces = data.get_ces(ce32.index(), ce32.len());
                                for &ce_ule in ces {
                                    self.pending.push(CollationElement::new_from_ule(ce_ule));
                                }
                                break 'ce32loop;
                            }
                            Tag::PrefixTag => {
                                let (default, mut trie) = data.get_default_and_trie(ce32.index());
                                ce32 = default;
                                for &ch in self.prefix.iter() {
                                    match trie.next(ch) {
                                        TrieResult::NoValue => {}
                                        TrieResult::NoMatch => {
                                            continue 'ce32loop;
                                        }
                                        TrieResult::Intermediate(ce32_i) => {
                                            ce32 = CollationElement32::new(ce32_i as u32);
                                        }
                                        TrieResult::FinalValue(ce32_i) => {
                                            ce32 = CollationElement32::new(ce32_i as u32);
                                            continue 'ce32loop;
                                        }
                                    }
                                }
                                continue 'ce32loop;
                            }
                            Tag::ContractionTag => {
                                let every_suffix_starts_with_combining =
                                    ce32.every_suffix_starts_with_combining();
                                let at_least_one_suffix_contains_starter =
                                    ce32.at_least_one_suffix_contains_starter();
                                let at_least_one_suffix_ends_with_non_starter =
                                    ce32.at_least_one_suffix_ends_with_non_starter();
                                let (default, mut trie) = data.get_default_and_trie(ce32.index());
                                ce32 = default;
                                if every_suffix_starts_with_combining
                                    && combining_characters.is_empty()
                                {
                                    continue 'ce32loop;
                                }
                                let mut longest_matching_state = trie.clone();
                                let mut longest_matching_index = 0;
                                let mut attempt = 0;
                                let mut i = 0;
                                let mut most_recent_skipped_ccc =
                                    CanonicalCombiningClass::NotReordered;
                                // XXX pending removals will in practice be small numbers.
                                // What if we made the item smaller than usize?
                                let mut pending_removals: SmallVec<[usize; 1]> = SmallVec::new();
                                while i < combining_characters.len() {
                                    let combining_and_class = &combining_characters[i];
                                    let ccc = combining_and_class.ccc();
                                    match (
                                        most_recent_skipped_ccc < ccc,
                                        trie.next(combining_and_class.character()),
                                    ) {
                                        (true, TrieResult::Intermediate(ce32_i)) => {
                                            let _ = combining_characters.remove(i);
                                            while let Some(idx) = pending_removals.pop() {
                                                combining_characters.remove(idx);
                                                i -= 1; // Adjust for the shortening
                                            }
                                            attempt = 0;
                                            longest_matching_index = i;
                                            longest_matching_state = trie.clone();
                                            ce32 = CollationElement32::new(ce32_i as u32);
                                        }
                                        (true, TrieResult::FinalValue(ce32_i)) => {
                                            let _ = combining_characters.remove(i);
                                            while let Some(idx) = pending_removals.pop() {
                                                combining_characters.remove(idx);
                                            }
                                            ce32 = CollationElement32::new(ce32_i as u32);
                                            continue 'ce32loop;
                                        }
                                        (_, TrieResult::NoValue) => {
                                            pending_removals.push(i);
                                            i += 1;
                                        }
                                        _ => {
                                            pending_removals.clear();
                                            most_recent_skipped_ccc = ccc;
                                            attempt += 1;
                                            i = longest_matching_index + attempt;
                                            trie = longest_matching_state.clone();
                                        }
                                    }
                                }
                                if !(at_least_one_suffix_contains_starter
                                    && combining_characters.is_empty())
                                {
                                    continue 'ce32loop;
                                }
                                debug_assert!(pending_removals.is_empty());
                                loop {
                                    let ahead = self.look_ahead(looked_ahead);
                                    looked_ahead += 1;
                                    if let Some(ch) = ahead {
                                        match trie.next(ch) {
                                            TrieResult::NoValue => {}
                                            TrieResult::NoMatch => {
                                                if !at_least_one_suffix_ends_with_non_starter {
                                                    continue 'ce32loop;
                                                }
                                                if !self
                                                    .decompositions
                                                    .decomposition_starts_with_non_starter
                                                    .contains(ch)
                                                {
                                                    continue 'ce32loop;
                                                }
                                                // The last-checked character is non-starter
                                                // and at least one contraction suffix ends
                                                // with a non-starter. Try a discontiguous
                                                // match.
                                                trie = longest_matching_state.clone();
                                                // For clarity, mint a new set of variables that
                                                // behave consistently with the
                                                // `combining_characters` case
                                                let mut longest_matching_index = 0;
                                                let mut attempt = 0;
                                                let mut i = 0;
                                                most_recent_skipped_ccc =
                                                    self.ccc.get(u32::from(ch));
                                                loop {
                                                    let ahead = self.look_ahead(looked_ahead + i);
                                                    if let Some(ch) = ahead {
                                                        let ccc = self.ccc.get(u32::from(ch));
                                                        match (
                                                            most_recent_skipped_ccc < ccc,
                                                            trie.next(ch),
                                                        ) {
                                                            (
                                                                true,
                                                                TrieResult::Intermediate(ce32_i),
                                                            ) => {
                                                                let _ = self
                                                                    .suffix
                                                                    .remove(looked_ahead + i);
                                                                while let Some(idx) =
                                                                    pending_removals.pop()
                                                                {
                                                                    self.suffix
                                                                        .remove(looked_ahead + idx);
                                                                    i -= 1; // Adjust for the shortening
                                                                }
                                                                attempt = 0;
                                                                longest_matching_index = i;
                                                                longest_matching_state =
                                                                    trie.clone();
                                                                ce32 = CollationElement32::new(
                                                                    ce32_i as u32,
                                                                );
                                                            }
                                                            (
                                                                true,
                                                                TrieResult::FinalValue(ce32_i),
                                                            ) => {
                                                                let _ = self
                                                                    .suffix
                                                                    .remove(looked_ahead + i);
                                                                while let Some(idx) =
                                                                    pending_removals.pop()
                                                                {
                                                                    self.suffix
                                                                        .remove(looked_ahead + idx);
                                                                }
                                                                ce32 = CollationElement32::new(
                                                                    ce32_i as u32,
                                                                );
                                                                continue 'ce32loop;
                                                            }
                                                            (_, TrieResult::NoValue) => {
                                                                pending_removals.push(i);
                                                                i += 1;
                                                            }
                                                            _ => {
                                                                pending_removals.clear();
                                                                most_recent_skipped_ccc = ccc;
                                                                attempt += 1;
                                                                i = longest_matching_index
                                                                    + attempt;
                                                                trie =
                                                                    longest_matching_state.clone();
                                                            }
                                                        }
                                                    } else {
                                                        continue 'ce32loop;
                                                    }
                                                }
                                            }
                                            TrieResult::Intermediate(ce32_i) => {
                                                longest_matching_state = trie.clone();
                                                drain_from_suffix = looked_ahead;
                                                ce32 = CollationElement32::new(ce32_i as u32);
                                            }
                                            TrieResult::FinalValue(ce32_i) => {
                                                drain_from_suffix = looked_ahead;
                                                ce32 = CollationElement32::new(ce32_i as u32);
                                                continue 'ce32loop;
                                            }
                                        }
                                    } else {
                                        continue 'ce32loop;
                                    }
                                }
                                // Unreachable
                            }
                            Tag::DigitTag => {
                                if self.numeric {
                                    let mut digits: SmallVec<[u8; 8]> = SmallVec::new(); // XXX figure out proper size
                                    digits.push(ce32.digit());
                                    let numeric_primary = 0xF000000u32; // XXX read from tailoring
                                    if combining_characters.is_empty() {
                                        // Numeric collation doesn't work with combining
                                        // characters applied to the digits.
                                        // XXX Does any tailoring actually tailor digits?
                                        while let Some(upcoming) = self.look_ahead(looked_ahead) {
                                            looked_ahead += 1;
                                            ce32 = self.tailoring.ce32_for_char(upcoming);
                                            if ce32 == FALLBACK_CE32 {
                                                ce32 = self.root.ce32_for_char(upcoming);
                                            }
                                            if ce32.tag_checked() != Some(Tag::DigitTag) {
                                                break;
                                            }
                                            drain_from_suffix = looked_ahead;
                                            digits.push(ce32.digit());
                                        }
                                    }
                                    // Skip leading zeros
                                    let mut zeros = 0;
                                    while let Some(&digit) = digits.get(zeros) {
                                        if digit != 0 {
                                            break;
                                        }
                                        zeros += 1;
                                    }
                                    if zeros == digits.len() {
                                        // All zeros, keep a zero
                                        zeros = digits.len() - 1;
                                    }
                                    let mut remaining = &digits[zeros..];
                                    while !remaining.is_empty() {
                                        // Numeric CEs are generated for segments of
                                        // up to 254 digits.
                                        let (head, tail) = if remaining.len() > 254 {
                                            remaining.split_at(254)
                                        } else {
                                            (remaining, &b""[..])
                                        };
                                        remaining = tail;
                                        // From ICU4C CollationIterator::appendNumericSegmentCEs
                                        if head.len() <= 7 {
                                            let mut digit_iter = head.iter();
                                            // Unwrap succeeds, because we always have at least one
                                            // digit to even start numeric processing.
                                            let mut value = u32::from(*digit_iter.next().unwrap());
                                            while let Some(&digit) = digit_iter.next() {
                                                value *= 10;
                                                value += u32::from(digit);
                                            }
                                            // Primary weight second byte values:
                                            //     74 byte values   2.. 75 for small numbers in two-byte primary weights.
                                            //     40 byte values  76..115 for medium numbers in three-byte primary weights.
                                            //     16 byte values 116..131 for large numbers in four-byte primary weights.
                                            //    124 byte values 132..255 for very large numbers with 4..127 digit pairs.
                                            let mut first_byte = 2u32;
                                            let mut num_bytes = 74u32;
                                            if value < num_bytes {
                                                self.pending.push(
                                                    CollationElement::new_from_primary(
                                                        numeric_primary
                                                            | ((first_byte + value) << 16),
                                                    ),
                                                );
                                                continue;
                                            }
                                            value -= num_bytes;
                                            first_byte += num_bytes;
                                            num_bytes = 40;
                                            if value < num_bytes * 254 {
                                                // Three-byte primary for 74..10233=74+40*254-1, good for year numbers and more.
                                                self.pending.push(
                                                    CollationElement::new_from_primary(
                                                        numeric_primary
                                                            | ((first_byte + value / 254) << 16)
                                                            | ((2 + value % 254) << 8),
                                                    ),
                                                );
                                                continue;
                                            }
                                            value -= num_bytes * 254;
                                            first_byte += num_bytes;
                                            num_bytes = 16;
                                            if value < num_bytes * 254 * 254 {
                                                // Four-byte primary for 10234..1042489=10234+16*254*254-1.
                                                let mut primary =
                                                    numeric_primary | (2 + value % 254);
                                                value /= 254;
                                                primary |= (2 + value % 254) << 8;
                                                value /= 254;
                                                primary |= (first_byte + value % 254) << 16;
                                                self.pending.push(
                                                    CollationElement::new_from_primary(primary),
                                                );
                                                continue;
                                            }
                                            // original value > 1042489
                                        }
                                        debug_assert!(head.len() >= 7);
                                        // The second primary byte value 132..255 indicates the number of digit pairs (4..127),
                                        // then we generate primary bytes with those pairs.
                                        // Omit trailing 00 pairs.
                                        // Decrement the value for the last pair.

                                        // Set the exponent. 4 pairs->132, 5 pairs->133, ..., 127 pairs->255.
                                        let mut len = head.len();
                                        let num_pairs = (len as u32 + 1) / 2; // as u32 OK, because capped to 254
                                        let mut primary =
                                            numeric_primary | ((132 - 4 + num_pairs) << 16);
                                        // Find the length without trailing 00 pairs.
                                        // XXX what guarantees [len - 2] not being index out of bounds?
                                        while head[len - 1] == 0 && head[len - 2] == 0 {
                                            len -= 2;
                                        }
                                        // Read the first pair
                                        let mut digit_iter = head[..len].iter();
                                        let mut pair = if len & 1 == 1 {
                                            // Only "half a pair" if we have an odd number of digits.
                                            u32::from(*digit_iter.next().unwrap())
                                        } else {
                                            u32::from(*digit_iter.next().unwrap()) * 10
                                                + u32::from(*digit_iter.next().unwrap())
                                        };
                                        pair = 11 + 2 * pair;
                                        let mut shift = 8u32;
                                        loop {
                                            let (left, right) =
                                                match (digit_iter.next(), digit_iter.next()) {
                                                    (Some(&left), Some(&right)) => (left, right),
                                                    _ => break,
                                                };
                                            if shift == 0 {
                                                primary |= pair;
                                                self.pending.push(
                                                    CollationElement::new_from_primary(primary),
                                                );
                                                primary = numeric_primary;
                                                shift = 16;
                                            } else {
                                                primary |= pair << shift;
                                                shift -= 8;
                                            }
                                            pair =
                                                11 + 2 * (u32::from(left) * 10 + u32::from(right));
                                        }
                                        primary |= (pair - 1) << shift;
                                        self.pending
                                            .push(CollationElement::new_from_primary(primary));
                                    }
                                    break 'ce32loop;
                                }
                                let ce32s = data.get_ce32s(ce32.index(), 1);
                                ce32 = CollationElement32::new_from_ule(ce32s[0]);
                                continue 'ce32loop;
                            }
                            // XXX how common are the following two cases? Should these
                            // be baked into the fast path, since they yield a single CE?
                            Tag::OffsetTag => {
                                self.pending.push(data.ce_from_offset_ce32(c, ce32));
                                break 'ce32loop;
                            }
                            Tag::ImplicitTag => {
                                self.pending
                                    .push(CollationElement::new_implicit_from_char(c));
                                break 'ce32loop;
                            }
                            Tag::FallbackTag
                            | Tag::ReservedTag3
                            | Tag::LongPrimaryTag
                            | Tag::LongSecondaryTag
                            | Tag::BuilderDataTag
                            | Tag::LeadSurrogateTag
                            | Tag::LatinExpansionTag
                            | Tag::U0000Tag
                            | Tag::HangulTag => {
                                unreachable!();
                            }
                        }
                    }
                }
                self.prefix_push(c);
                debug_assert!(drain_from_suffix == 0 || combining_characters.is_empty());
                let mut i = 0;
                'combining: while i < combining_characters.len() {
                    c = combining_characters[i].character();
                    let diacritic_index = (c as usize).wrapping_sub(COMBINING_DIACRITICS_BASE);
                    if let Some(&diacritic) = self.diacritics.get(diacritic_index) {
                        // TODO: unlikely annotation for the first two conditions here:
                        if c == '\u{0307}'
                            && self.lithuanian_dot_above
                            && i + 1 < combining_characters.len()
                        {
                            let next_c = combining_characters[i + 1].character();
                            if next_c == '\u{0300}' || next_c == '\u{0301}' || next_c == '\u{0303}'
                            {
                                // Lithuanian contracts COMBINING DOT ABOVE with three other diacritics of the
                                // same combining class such that the COMBINING DOT ABOVE is ignored for
                                // collation. Since the combining class is the same, it's valid to simply
                                // look at the next character in `combining_characters`.
                                i += 1;
                                continue 'combining;
                            }
                        }
                        // Unwrap: expectation of data integrity
                        self.pending.push(
                            CollationElement32::new_from_ule(diacritic)
                                .to_ce_self_contained()
                                .unwrap(),
                        );
                        self.mark_prefix_unmatchable();
                        i += 1;
                        continue 'combining;
                    }
                    let _ = combining_characters.drain(..=i);
                    data = self.tailoring;
                    ce32 = data.ce32_for_char(c);
                    if ce32 == FALLBACK_CE32 {
                        data = self.root;
                        ce32 = data.ce32_for_char(c);
                    }
                    continue 'outer;
                }
                // XXX the borrow checker didn't like the iterator formulation
                // for the loop below, because the `Drain` would have kept `self`
                // mutable borrowed when trying to call `prefix_push`.
                i = 0;
                while i < drain_from_suffix {
                    let ch = self.suffix[i];
                    self.prefix_push(ch);
                    i += 1;
                }
                let _ = self.suffix.drain(..drain_from_suffix);
                if self.suffix.is_empty() {
                    if let Some(c) = self.iter.next() {
                        self.suffix.push(c);
                    }
                }
                return self.pending.remove(0);
            }
        } else {
            NO_CE
        }
    }
}

#[derive(Eq, PartialEq, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Strength {
    Primary = 0,
    Secondary = 1,
    Tertiary = 2,
    Quaternary = 3,
    Identical = 4,
}

#[derive(Eq, PartialEq, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum AlternateHandling {
    NonIgnorable = 0,
    Shifted = 1,
    // Possible future values: ShiftTrimmed, Blanked
}

#[derive(Eq, PartialEq, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum CaseFirst {
    Off = 0,
    LowerFirst = 1,
    UpperFirst = 2,
}

#[derive(Eq, PartialEq)]
#[repr(u8)]
pub enum MaxVariable {
    Space = 0,
    Punctuation = 1,
    Symbol = 2,
    Currency = 3,
}

/// Options settable by the user of the API.
///
/// See https://www.unicode.org/reports/tr35/tr35-collation.html#Setting_Options
///
/// # Options
///
/// ## Strength
///
/// This is the BCP47 key `ks`. The default is `Strength::Tertiary`.
///
/// ## Alternate Handling
///
/// This is the BCP47 key `ka`. Note that `ShiftTrimmed` and `Blanked` are
/// unimplemented. The default is `AlternateHandling::NonIgnorable`.
///
/// Note: Thai forces this to `AlternateHandling::Shifted` regardless of the
/// option given here.
///
/// ## Case Level
///
/// See https://www.unicode.org/reports/tr35/tr35-collation.html#Case_Parameters
/// This is the BCP47 key `kc`. The default is `false` (off).
///
/// ## Case First
///
/// See https://www.unicode.org/reports/tr35/tr35-collation.html#Case_Parameters
/// This is the BCP47 key `kf`. Three possibilities: `CaseFirst::Off` (default),
/// `CaseFirst::Lower`, and `CaseFirst::Upper`.
///
/// Note: Danish and Maltese force this to `CaseFirst::Upper` regardless of the
/// option given here.
///
/// ## Backward second level
///
/// Compare the second level in backward order. This is the BCP47 key `kb`. `kb`
/// is prohibited by ECMA 402.
///
/// # Numeric
///
/// This is the BCP47 key `kn`. When set to `true` (on), any sequence of decimal
/// digits (General_Category = Nd) is sorted at a primary level accoding to the
/// numeric value. The default is `false` (off).
///
/// # Unsupported BCP47 options
///
/// Reordering (BCP47 `kr`) currently cannot be set via the API and is implied
/// by the locale of the collation. `kr` is probihibited by ECMA 402.
///
/// Normalization is always enabled and cannot be turned off. Therefore, there
/// is no option corresponding to BCP47 `kk`. `kk` is prohibited by ECMA 402.
///
/// Hiragana quaternary handling is part of the strength. The BCP47 key `kh`
/// is unsupported. `kh` is deprecated and prohibited by ECMA 402.
///
/// Variable top (BCP47 `vt`) is unsupported (use Max Variable instead). `vt`
/// is deprecated and prohibited by ECMA 402.
#[derive(Copy, Clone, Debug)]
pub struct CollatorOptions(u32);

impl CollatorOptions {
    /// Bits 0..2 : Strength
    const STRENGTH_MASK: u32 = 0b111;
    /// Bits 3..4 : Alternate handling: 00 non-ignorable, 01 shifted,
    ///             10 reserved for shift-trimmed, 11 reserved for blanked.
    ///             In other words, bit 4 is currently always 0.
    const ALTERNATE_HANDLING_MASK: u32 = 1 << 3;
    /// Bits 5..6 : 2-bit max variable value to be shifted by `MAX_VARIABLE_SHIFT`.
    const MAX_VARIABLE_MASK: u32 = 0b01100000;
    const MAX_VARIABLE_SHIFT: u32 = 5;
    /// Bit     7 : Reserved for extending max variable.
    /// Bit     8 : Sort uppercase first if case level or case first is on.
    const UPPER_FIRST_MASK: u32 = 1 << 8;
    /// Bit     9 : Keep the case bits in the tertiary weight (they trump
    ///             other tertiary values)
    ///             unless case level is on (when they are *moved* into the separate case level).
    ///             By default, the case bits are removed from the tertiary weight (ignored).
    ///             When CASE_FIRST is off, UPPER_FIRST must be off too, corresponding to
    ///             the tri-value UCOL_CASE_FIRST attribute: UCOL_OFF vs. UCOL_LOWER_FIRST vs.
    ///             UCOL_UPPER_FIRST.
    const CASE_FIRST_MASK: u32 = 1 << 9;
    /// Bit    10 : Insert the case level between the secondary and tertiary levels.
    const CASE_LEVEL_MASK: u32 = 1 << 10;
    /// Bit    11 : Backward secondary level
    const BACKWARD_SECOND_LEVEL_MASK: u32 = 1 << 11;
    /// Bit    12 : Numeric
    const NUMERIC_MASK: u32 = 1 << 12;

    /// Whether strength is explicitly set.
    const EXPLICIT_STRENGTH_MASK: u32 = 1 << 31;
    /// Whether max variable is explicitly set.
    const EXPLICIT_MAX_VARIABLE_MASK: u32 = 1 << 30;
    /// Whether alternate handling is explicitly set.
    const EXPLICIT_ALTERNATE_HANDLING_MASK: u32 = 1 << 29;
    /// Whether case level is explicitly set.
    const EXPLICIT_CASE_LEVEL_MASK: u32 = 1 << 28;
    /// Whether case first is explicitly set.
    const EXPLICIT_CASE_FIRST_MASK: u32 = 1 << 27;
    /// Whether backward secondary is explicitly set.
    const EXPLICIT_BACKWARD_SECOND_LEVEL_MASK: u32 = 1 << 26;
    /// Whether numeric is explicitly set.
    const EXPLICIT_NUMERIC_MASK: u32 = 1 << 25;

    pub const fn new() -> Self {
        Self(Strength::Tertiary as u32)
    }

    pub fn strength(&self) -> Strength {
        let mut bits = self.0 & CollatorOptions::STRENGTH_MASK;
        if bits > 4 {
            debug_assert!(false, "Bad value for strength");
            bits = 4;
        }
        // By construction in range and, therefore,
        // never UB.
        unsafe { core::mem::transmute(bits as u8) }
    }

    pub fn set_strength(&mut self, strength: Option<Strength>) {
        self.0 &= !CollatorOptions::STRENGTH_MASK;
        if let Some(strength) = strength {
            self.0 |= CollatorOptions::EXPLICIT_STRENGTH_MASK;
            self.0 |= strength as u32;
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_STRENGTH_MASK;
        }
    }

    pub fn max_variable(&self) -> MaxVariable {
        unsafe {
            core::mem::transmute(
                ((self.0 & CollatorOptions::MAX_VARIABLE_MASK)
                    >> CollatorOptions::MAX_VARIABLE_SHIFT) as u8,
            )
        }
    }

    pub fn set_max_variable(&mut self, max_variable: Option<MaxVariable>) {
        self.0 &= !CollatorOptions::MAX_VARIABLE_MASK;
        if let Some(max_variable) = max_variable {
            self.0 |= CollatorOptions::EXPLICIT_MAX_VARIABLE_MASK;
            self.0 |= (max_variable as u32) << CollatorOptions::MAX_VARIABLE_SHIFT;
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_MAX_VARIABLE_MASK;
        }
    }

    pub fn alternate_handling(&self) -> AlternateHandling {
        if (self.0 & CollatorOptions::ALTERNATE_HANDLING_MASK) != 0 {
            AlternateHandling::Shifted
        } else {
            AlternateHandling::NonIgnorable
        }
    }

    pub fn set_alternate_handling(&mut self, alternate_handling: Option<AlternateHandling>) {
        self.0 &= !CollatorOptions::ALTERNATE_HANDLING_MASK;
        if let Some(alternate_handling) = alternate_handling {
            self.0 |= CollatorOptions::EXPLICIT_ALTERNATE_HANDLING_MASK;
            if alternate_handling == AlternateHandling::Shifted {
                self.0 |= CollatorOptions::ALTERNATE_HANDLING_MASK;
            }
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_ALTERNATE_HANDLING_MASK;
        }
    }

    pub fn case_level(&self) -> bool {
        (self.0 & CollatorOptions::CASE_LEVEL_MASK) != 0
    }

    pub fn set_case_level(&mut self, case_level: Option<bool>) {
        self.0 &= !CollatorOptions::CASE_LEVEL_MASK;
        if let Some(case_level) = case_level {
            self.0 |= CollatorOptions::EXPLICIT_CASE_LEVEL_MASK;
            if case_level {
                self.0 |= CollatorOptions::ALTERNATE_HANDLING_MASK;
            }
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_CASE_LEVEL_MASK;
        }
    }

    pub fn case_first(&self) -> CaseFirst {
        if (self.0 & CollatorOptions::CASE_FIRST_MASK) != 0 {
            if (self.0 & CollatorOptions::UPPER_FIRST_MASK) != 0 {
                CaseFirst::UpperFirst
            } else {
                CaseFirst::LowerFirst
            }
        } else {
            CaseFirst::Off
        }
    }

    pub fn set_case_first(&mut self, case_first: Option<CaseFirst>) {
        self.0 &= !(CollatorOptions::CASE_FIRST_MASK | CollatorOptions::UPPER_FIRST_MASK);
        if let Some(case_first) = case_first {
            self.0 |= CollatorOptions::EXPLICIT_CASE_FIRST_MASK;
            match case_first {
                CaseFirst::Off => {}
                CaseFirst::LowerFirst => {
                    self.0 |= CollatorOptions::CASE_FIRST_MASK;
                }
                CaseFirst::UpperFirst => {
                    self.0 |= CollatorOptions::CASE_FIRST_MASK;
                    self.0 |= CollatorOptions::UPPER_FIRST_MASK;
                }
            }
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_CASE_FIRST_MASK;
        }
    }

    pub fn backward_second_level(&self) -> bool {
        (self.0 & CollatorOptions::BACKWARD_SECOND_LEVEL_MASK) != 0
    }

    pub fn set_backward_second_level(&mut self, backward_second_level: Option<bool>) {
        self.0 &= !CollatorOptions::BACKWARD_SECOND_LEVEL_MASK;
        if let Some(backward_second_level) = backward_second_level {
            self.0 |= CollatorOptions::EXPLICIT_BACKWARD_SECOND_LEVEL_MASK;
            if backward_second_level {
                self.0 |= CollatorOptions::BACKWARD_SECOND_LEVEL_MASK;
            }
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_BACKWARD_SECOND_LEVEL_MASK;
        }
    }

    pub fn numeric(&self) -> bool {
        (self.0 & CollatorOptions::NUMERIC_MASK) != 0
    }

    pub fn set_numeric(&mut self, numeric: Option<bool>) {
        self.0 &= !CollatorOptions::NUMERIC_MASK;
        if let Some(numeric) = numeric {
            self.0 |= CollatorOptions::EXPLICIT_NUMERIC_MASK;
            if numeric {
                self.0 |= CollatorOptions::NUMERIC_MASK;
            }
        } else {
            self.0 &= !CollatorOptions::EXPLICIT_NUMERIC_MASK;
        }
    }

    // If strength is <= secondary, returns `None`.
    // Otherwise, returns the appropriate mask.
    pub(crate) fn tertiary_mask(&self) -> Option<u16> {
        if self.strength() <= Strength::Secondary {
            None
        } else if (self.0 & (CollatorOptions::CASE_FIRST_MASK | CollatorOptions::CASE_LEVEL_MASK))
            == CollatorOptions::CASE_FIRST_MASK
        {
            Some(CASE_MASK | TERTIARY_MASK)
        } else {
            Some(TERTIARY_MASK)
        }
    }

    pub(crate) fn upper_first(&self) -> bool {
        (self.0 & CollatorOptions::UPPER_FIRST_MASK) != 0
    }

    pub fn set_defaults(&mut self, other: CollatorOptions) {
        if self.0 & CollatorOptions::EXPLICIT_STRENGTH_MASK == 0 {
            self.0 &= !CollatorOptions::STRENGTH_MASK;
            self.0 |= other.0 & CollatorOptions::STRENGTH_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_STRENGTH_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_MAX_VARIABLE_MASK == 0 {
            self.0 &= !CollatorOptions::MAX_VARIABLE_MASK;
            self.0 |= other.0 & CollatorOptions::MAX_VARIABLE_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_MAX_VARIABLE_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_ALTERNATE_HANDLING_MASK == 0 {
            self.0 &= !CollatorOptions::ALTERNATE_HANDLING_MASK;
            self.0 |= other.0 & CollatorOptions::ALTERNATE_HANDLING_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_ALTERNATE_HANDLING_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_CASE_LEVEL_MASK == 0 {
            self.0 &= !CollatorOptions::CASE_LEVEL_MASK;
            self.0 |= other.0 & CollatorOptions::CASE_LEVEL_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_CASE_LEVEL_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_CASE_FIRST_MASK == 0 {
            self.0 &= !(CollatorOptions::CASE_FIRST_MASK | CollatorOptions::UPPER_FIRST_MASK);
            self.0 |=
                other.0 & (CollatorOptions::CASE_FIRST_MASK | CollatorOptions::UPPER_FIRST_MASK);
            self.0 |= other.0 & CollatorOptions::EXPLICIT_CASE_FIRST_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_BACKWARD_SECOND_LEVEL_MASK == 0 {
            self.0 &= !CollatorOptions::BACKWARD_SECOND_LEVEL_MASK;
            self.0 |= other.0 & CollatorOptions::BACKWARD_SECOND_LEVEL_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_BACKWARD_SECOND_LEVEL_MASK;
        }
        if self.0 & CollatorOptions::EXPLICIT_NUMERIC_MASK == 0 {
            self.0 &= !CollatorOptions::NUMERIC_MASK;
            self.0 |= other.0 & CollatorOptions::NUMERIC_MASK;
            self.0 |= other.0 & CollatorOptions::EXPLICIT_NUMERIC_MASK;
        }
    }
}

struct AnyQuaternaryAccumulator(u32);

impl AnyQuaternaryAccumulator {
    #[inline(always)]
    pub fn new() -> Self {
        AnyQuaternaryAccumulator(0)
    }
    #[inline(always)]
    pub fn accumulate(&mut self, non_primary: NonPrimary) {
        self.0 |= non_primary.0
    }
    #[inline(always)]
    pub fn has_quaternary(&self) -> bool {
        self.0 & u32::from(QUATERNARY_MASK) != 0
    }
}

// Hoisted to function, because the compiler doesn't like having
// to identical closures.
#[inline(always)]
fn utf16_error_to_replacement(r: Result<char, DecodeUtf16Error>) -> char {
    r.unwrap_or(REPLACEMENT_CHARACTER)
}

pub struct Collator {
    root: DataPayload<CollationDataV1Marker>,
    tailoring: Option<DataPayload<CollationDataV1Marker>>,
    jamo: DataPayload<CollationJamoV1Marker>,
    diacritics: DataPayload<CollationDiacriticsV1Marker>,
    options: CollatorOptions,
    reordering: Option<DataPayload<CollationReorderingV1Marker>>,
    decompositions: DataPayload<CanonicalDecompositionDataV1Marker>,
    ccc: DataPayload<UnicodePropertyMapV1Marker<CanonicalCombiningClass>>,
    lithuanian_dot_above: bool,
}

impl<'data> Collator {
    pub fn try_new<T: Into<Locale>, D>(
        locale: T,
        data_provider: &D,
        options: CollatorOptions,
    ) -> Result<Self, CollatorError>
    where
        D: ResourceProvider<CollationDataV1Marker>
            + ResourceProvider<CollationDiacriticsV1Marker>
            + ResourceProvider<CollationJamoV1Marker>
            + ResourceProvider<CollationMetadataV1Marker>
            + ResourceProvider<CollationReorderingV1Marker>
            + ResourceProvider<CanonicalDecompositionDataV1Marker>
            + DynProvider<
                icu_properties::provider::UnicodePropertyMapV1Marker<
                    icu_properties::CanonicalCombiningClass,
                >,
            > + ?Sized,
    {
        let locale: Locale = locale.into();
        let resource_options: ResourceOptions = locale.into();

        let metadata_payload: DataPayload<crate::provider::CollationMetadataV1Marker> =
            data_provider
                .load_resource(&DataRequest {
                    options: resource_options.clone(),
                    metadata: Default::default(),
                })?
                .take_payload()?;

        let metadata = metadata_payload.get();

        let tailoring: Option<DataPayload<crate::provider::CollationDataV1Marker>> =
            if metadata.tailored() {
                Some(
                    data_provider
                        .load_resource(&DataRequest {
                            options: resource_options.clone(),
                            metadata: Default::default(),
                        })?
                        .take_payload()?,
                )
            } else {
                None
            };

        let reordering: Option<DataPayload<crate::provider::CollationReorderingV1Marker>> =
            if metadata.reordering() {
                Some(
                    data_provider
                        .load_resource(&DataRequest {
                            options: resource_options.clone(),
                            metadata: Default::default(),
                        })?
                        .take_payload()?,
                )
            } else {
                None
            };

        if let Some(reordering) = &reordering {
            if reordering.get().reorder_table.len() != 256 {
                return Err(CollatorError::MalformedData);
            }
        }

        let root: DataPayload<CollationDataV1Marker> = data_provider
            .load_resource(&DataRequest::default())?
            .take_payload()?;

        let diacritics: DataPayload<CollationDiacriticsV1Marker> = data_provider
            .load_resource(&DataRequest {
                options: if metadata.tailored_diacritics() {
                    resource_options.clone()
                    } else {
                        ResourceOptions::default()
                    },
                metadata: Default::default(),
            })?
            .take_payload()?;

        if diacritics.get().ce32s.len() != COMBINING_DIACRITICS_COUNT {
            return Err(CollatorError::MalformedData);
        }

        let jamo: DataPayload<CollationJamoV1Marker> = data_provider
            .load_resource(&DataRequest::default())? // TODO: load other jamo tables
            .take_payload()?;

        if jamo.get().ce32s.len() != JAMO_COUNT {
            return Err(CollatorError::MalformedData);
        }

        let decompositions: DataPayload<CanonicalDecompositionDataV1Marker> = data_provider
            .load_resource(&DataRequest::default())?
            .take_payload()?;

        let ccc: DataPayload<UnicodePropertyMapV1Marker<CanonicalCombiningClass>> =
            icu_properties::maps::get_canonical_combining_class(data_provider)?;

        let mut altered_defaults = CollatorOptions::new();

        if metadata.alternate_shifted() {
            altered_defaults.set_alternate_handling(Some(AlternateHandling::Shifted));
        }

        altered_defaults.set_case_first(Some(metadata.case_first()));
        altered_defaults.set_max_variable(Some(metadata.max_variable()));

        let mut merged_options = options.clone();
        merged_options.set_defaults(altered_defaults);

        Ok(Collator {
            root: root,
            tailoring: tailoring,
            jamo: jamo,
            diacritics: diacritics,
            options: merged_options,
            reordering: reordering,
            decompositions: decompositions,
            ccc: ccc,
            lithuanian_dot_above: metadata.lithuanian_dot_above(),
        })
    }

    pub fn compare_utf16(&self, left: &[u16], right: &[u16]) -> Ordering {
        // XXX u16-compare prefix, but some knowledge of possible
        // PrefixTag cases is needed to know how much to back off
        // before real collation to feed the prefix into the
        // collation unit lookup.
        // ICU4C says this, which suggests its backwardUnsafe set
        // doesn't work exactly here unless we know the worst-case
        // prefix length and pre-normalize that many characters into
        // the prefix buffer of CollationElements:
        //
        // "Pass the actual start of each string into the CollationIterators,
        // plus the equalPrefixLength position,
        // so that prefix matches back into the equal prefix work."
        let ret = self.compare_impl(
            decode_utf16(left.iter().copied()).map(utf16_error_to_replacement),
            decode_utf16(right.iter().copied()).map(utf16_error_to_replacement),
        );
        if self.options.strength() == Strength::Identical && ret == Ordering::Equal {
            return Decomposition::new(
                decode_utf16(left.iter().copied()).map(utf16_error_to_replacement),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            )
            .cmp(Decomposition::new(
                decode_utf16(right.iter().copied()).map(utf16_error_to_replacement),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            ));
        }
        ret
    }

    pub fn compare(&self, left: &str, right: &str) -> Ordering {
        // XXX byte-compare prefix, but some knowledge of possible
        // PrefixTag cases is needed to know how much to back off
        // before real collation to feed the prefix into the
        // collation unit lookup.
        // ICU4C says this, which suggests its backwardUnsafe set
        // doesn't work exactly here unless we know the worst-case
        // prefix length and pre-normalize that many characters into
        // the prefix buffer of CollationElements:
        //
        // "Pass the actual start of each string into the CollationIterators,
        // plus the equalPrefixLength position,
        // so that prefix matches back into the equal prefix work."
        let ret = self.compare_impl(left.chars(), right.chars());
        if self.options.strength() == Strength::Identical && ret == Ordering::Equal {
            return Decomposition::new(
                left.chars(),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            )
            .cmp(Decomposition::new(
                right.chars(),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            ));
        }
        ret
    }

    pub fn compare_utf8(&self, left: &[u8], right: &[u8]) -> Ordering {
        // XXX byte-compare prefix, but some knowledge of possible
        // PrefixTag cases is needed to know how much to back off
        // before real collation to feed the prefix into the
        // collation unit lookup.
        // ICU4C says this, which suggests its backwardUnsafe set
        // doesn't work exactly here unless we know the worst-case
        // prefix length and pre-normalize that many characters into
        // the prefix buffer of CollationElements:
        //
        // "Pass the actual start of each string into the CollationIterators,
        // plus the equalPrefixLength position,
        // so that prefix matches back into the equal prefix work."
        let ret = self.compare_impl(left.chars(), right.chars());
        if self.options.strength() == Strength::Identical && ret == Ordering::Equal {
            return Decomposition::new(
                left.chars(),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            )
            .cmp(Decomposition::new(
                right.chars(),
                self.decompositions.get(),
                &self.ccc.get().code_point_trie,
            ));
        }
        ret
    }

    fn compare_impl<I: Iterator<Item = char>>(&self, left_chars: I, right_chars: I) -> Ordering {
        let tailoring: &DataPayload<CollationDataV1Marker> =
            if let Some(tailoring) = &self.tailoring {
                tailoring
            } else {
                // If the root collation is valid for the locale,
                // use the root as the tailoring so that reads from the
                // tailoring always succeed.
                //
                // XXX Do we instead want to have an untailored
                // copypaste of the iterator that omits the tailoring
                // branches for performance at the expense of code size
                // and having to maintain both a tailoring-capable and
                // a tailoring-incapable version of the iterator?
                // Or, in order not to flip the branch prediction around,
                // should we have a no-op tailoring that contains a
                // specially-crafted CodePointTrie that always returns
                // a FALLBACK_CE32 after a single branch?
                &self.root
            };

        // Sadly, it looks like variable CEs require us to store the full
        // 64-bit CEs instead of storing only the NonPrimary part.
        // XXX Consider having to flavors of this method:
        // one that can deal with variables shifted to quaternary
        // and another that doesn't support that.
        // XXX what about primary ignorables with primary + case?

        // XXX figure out a proper size for these
        let mut left_ces: SmallVec<[CollationElement; 8]> = SmallVec::new();
        let mut right_ces: SmallVec<[CollationElement; 8]> = SmallVec::new();

        // The algorithm comes from CollationCompare::compareUpToQuaternary in ICU4C.

        let mut any_variable = false;
        let variable_top = if self.options.alternate_handling() == AlternateHandling::NonIgnorable {
            0
        } else {
            // +1 so that we can use "<" and primary ignorables test out early.
            tailoring
                .get()
                .last_primary_for_group(self.options.max_variable())
                + 1
        };

        let mut left = CollationElements::new(
            left_chars,
            self.root.get(),
            tailoring.get(),
            <&[<u32 as AsULE>::ULE; JAMO_COUNT]>::try_from(self.jamo.get().ce32s.as_ule_slice())
                .unwrap(), // length already validated
            <&[<u32 as AsULE>::ULE; COMBINING_DIACRITICS_COUNT]>::try_from(
                self.diacritics.get().ce32s.as_ule_slice(),
            )
            .unwrap(), // length already validated
            self.decompositions.get(),
            &self.ccc.get().code_point_trie,
            self.options.numeric(),
            self.lithuanian_dot_above,
        );
        let mut right = CollationElements::new(
            right_chars,
            self.root.get(),
            tailoring.get(),
            <&[<u32 as AsULE>::ULE; JAMO_COUNT]>::try_from(self.jamo.get().ce32s.as_ule_slice())
                .unwrap(), // length already validated
            <&[<u32 as AsULE>::ULE; COMBINING_DIACRITICS_COUNT]>::try_from(
                self.diacritics.get().ce32s.as_ule_slice(),
            )
            .unwrap(), // length already validated
            self.decompositions.get(),
            &self.ccc.get().code_point_trie,
            self.options.numeric(),
            self.lithuanian_dot_above,
        );
        loop {
            let mut left_primary;
            'left_primary_loop: loop {
                let ce = left.next();
                left_primary = ce.primary();
                // XXX consider compiling out the variable handling when we know we aren't
                // shifting variable CEs.
                if !(left_primary < variable_top && left_primary > MERGE_SEPARATOR_PRIMARY) {
                    left_ces.push(ce);
                } else {
                    // Variable CE, shift it to quaternary level.
                    // Ignore all following primary ignorables, and shift further variable CEs.
                    any_variable = true;
                    // Relative to ICU4C, the next line is hoisted out of the following loop
                    // in order to keep the variables called `ce` immutable to make it easier
                    // to reason about each assignment into `ce` resulting in exactly a single
                    // push into `left_ces`.
                    left_ces.push(ce.clone_with_non_primary_zeroed());
                    loop {
                        // This loop is simpler than in ICU4C; unlike in C++, we get to break by label.
                        let ce = left.next();
                        left_primary = ce.primary();
                        if left_primary != 0
                            && !(left_primary < variable_top
                                && left_primary > MERGE_SEPARATOR_PRIMARY)
                        {
                            // Neither a primary ignorable nor a variable CE.
                            left_ces.push(ce);
                            break 'left_primary_loop;
                        }
                        // If `left_primary == 0`, the following line ignores a primary-ignorable.
                        // Otherwise, it shifts a variable CE.
                        left_ces.push(ce.clone_with_non_primary_zeroed());
                    }
                }
                if left_primary != 0 {
                    break;
                }
            }
            let mut right_primary;
            'right_primary_loop: loop {
                let ce = right.next();
                right_primary = ce.primary();
                // XXX consider compiling out the variable handling when we know we aren't
                // shifting variable CEs.
                if !(right_primary < variable_top && right_primary > MERGE_SEPARATOR_PRIMARY) {
                    right_ces.push(ce);
                } else {
                    // Variable CE, shift it to quaternary level.
                    // Ignore all following primary ignorables, and shift further variable CEs.
                    any_variable = true;
                    // Relative to ICU4C, the next line is hoisted out of the following loop
                    // in order to keep the variables called `ce` immutable to make it easier
                    // to reason about each assignment into `ce` resulting in exactly a single
                    // push into `right_ces`.
                    right_ces.push(ce.clone_with_non_primary_zeroed());
                    loop {
                        // This loop is simpler than in ICU4C; unlike in C++, we get to break by label.
                        let ce = right.next();
                        right_primary = ce.primary();
                        if right_primary != 0
                            && !(right_primary < variable_top
                                && right_primary > MERGE_SEPARATOR_PRIMARY)
                        {
                            // Neither a primary ignorable nor a variable CE.
                            right_ces.push(ce);
                            break 'right_primary_loop;
                        }
                        // If `right_primary == 0`, the following line ignores a primary-ignorable.
                        // Otherwise, it shifts a variable CE.
                        right_ces.push(ce.clone_with_non_primary_zeroed());
                    }
                }
                if right_primary != 0 {
                    break;
                }
            }
            if left_primary != right_primary {
                if let Some(reordering) = &self.reordering {
                    left_primary = reordering.get().reorder(left_primary);
                    right_primary = reordering.get().reorder(right_primary);
                }
                if left_primary < right_primary {
                    return Ordering::Less;
                }
                return Ordering::Greater;
            }
            if left_primary == NO_CE_PRIMARY {
                break;
            }
        }

        // Sadly, we end up pushing the sentinel value, which means these
        // `SmallVec`s allocate more often than if we didn't actually
        // store the sentinel.
        debug_assert_eq!(left_ces[left_ces.len() - 1], NO_CE);
        debug_assert_eq!(right_ces[right_ces.len() - 1], NO_CE);

        // Note: unwrap_or_default in the iterations below should never
        // actually end up using the "_or_default" part, because the sentinel
        // is in the vectors? These could be changed to `unwrap()` if we
        // preferred panic in case of a bug.
        // XXX: Should we save one slot by not putting the sentinel in the
        // vectors? So far, the answer seems "no", as it would complicate
        // the primary comparison above.

        // Compare the buffered secondary & tertiary weights.
        // We might skip the secondary level but continue with the case level
        // which is turned on separately.
        if self.options.strength() >= Strength::Secondary {
            if !self.options.backward_second_level() {
                let mut left_iter = left_ces.iter();
                let mut right_iter = right_ces.iter();
                let mut left_secondary;
                let mut right_secondary;
                loop {
                    loop {
                        left_secondary = left_iter.next().unwrap_or_default().secondary();
                        if left_secondary != 0 {
                            break;
                        }
                    }
                    loop {
                        right_secondary = right_iter.next().unwrap_or_default().secondary();
                        if right_secondary != 0 {
                            break;
                        }
                    }
                    if left_secondary != right_secondary {
                        if left_secondary < right_secondary {
                            return Ordering::Less;
                        }
                        return Ordering::Greater;
                    }
                    if left_secondary == NO_CE_SECONDARY {
                        break;
                    }
                }
            } else {
                let mut left_iter = left_ces.iter();
                let mut right_iter = right_ces.iter();

                // Skip the in-data sentinel.
                let no_ce = left_iter.next_back();
                debug_assert_eq!(no_ce, Some(&NO_CE));
                let no_ce = right_iter.next_back();
                debug_assert_eq!(no_ce, Some(&NO_CE));

                let mut left_secondary;
                let mut right_secondary;
                loop {
                    loop {
                        left_secondary = left_iter.next_back().unwrap_or(&NO_CE).secondary();
                        if left_secondary != 0 {
                            break;
                        }
                    }
                    loop {
                        right_secondary = right_iter.next_back().unwrap_or(&NO_CE).secondary();
                        if right_secondary != 0 {
                            break;
                        }
                    }
                    if left_secondary != right_secondary {
                        if left_secondary < right_secondary {
                            return Ordering::Less;
                        }
                        return Ordering::Greater;
                    }
                    if left_secondary == NO_CE_SECONDARY {
                        break;
                    }
                }
            }
        }

        if self.options.case_level() {
            if self.options.strength() == Strength::Primary {
                // Primary+caseLevel: Ignore case level weights of primary ignorables.
                // Otherwise we would get a-umlaut > a
                // which is not desirable for accent-insensitive sorting.
                // Check for (lower 32 bits) == 0 as well because variable CEs are stored
                // with only primary weights.
                let mut left_non_primary;
                let mut right_non_primary;
                let mut left_case;
                let mut right_case;
                let mut left_iter = left_ces.iter();
                let mut right_iter = right_ces.iter();
                loop {
                    loop {
                        let ce = left_iter.next().unwrap_or_default();
                        left_non_primary = ce.non_primary();
                        if !ce.either_half_zero() {
                            break;
                        }
                    }
                    left_case = left_non_primary.case();
                    loop {
                        let ce = right_iter.next().unwrap_or_default();
                        right_non_primary = ce.non_primary();
                        if !ce.either_half_zero() {
                            break;
                        }
                    }
                    right_case = right_non_primary.case();
                    // No need to handle NO_CE and MERGE_SEPARATOR specially:
                    // There is one case weight for each previous-level weight,
                    // so level length differences were handled there.
                    if left_case != right_case {
                        if !self.options.upper_first() {
                            if left_case < right_case {
                                return Ordering::Less;
                            }
                            return Ordering::Greater;
                        }
                        if left_case < right_case {
                            return Ordering::Greater;
                        }
                        return Ordering::Less;
                    }
                    if left_non_primary.secondary() == NO_CE_SECONDARY {
                        break;
                    }
                }
            } else {
                // Secondary+caseLevel: By analogy with the above,
                // ignore case level weights of secondary ignorables.
                //
                // Note: A tertiary CE has uppercase case bits (0.0.ut)
                // to keep tertiary+caseFirst well-formed.
                //
                // Tertiary+caseLevel: Also ignore case level weights of secondary ignorables.
                // Otherwise a tertiary CE's uppercase would be no greater than
                // a primary/secondary CE's uppercase.
                // (See UCA well-formedness condition 2.)
                // We could construct a special case weight higher than uppercase,
                // but it's simpler to always ignore case weights of secondary ignorables,
                // turning 0.0.ut into 0.0.0.t.
                // (See LDML Collation, Case Parameters.)
                let mut left_non_primary;
                let mut right_non_primary;
                let mut left_case;
                let mut right_case;
                let mut left_iter = left_ces.iter();
                let mut right_iter = right_ces.iter();
                loop {
                    loop {
                        left_non_primary = left_iter.next().unwrap_or_default().non_primary();
                        if left_non_primary.secondary() != 0 {
                            break;
                        }
                    }
                    left_case = left_non_primary.case();
                    loop {
                        right_non_primary = right_iter.next().unwrap_or_default().non_primary();
                        if right_non_primary.secondary() != 0 {
                            break;
                        }
                    }
                    right_case = right_non_primary.case();
                    // No need to handle NO_CE and MERGE_SEPARATOR specially:
                    // There is one case weight for each previous-level weight,
                    // so level length differences were handled there.
                    if left_case != right_case {
                        if !self.options.upper_first() {
                            if left_case < right_case {
                                return Ordering::Less;
                            }
                            return Ordering::Greater;
                        }
                        if left_case < right_case {
                            return Ordering::Greater;
                        }
                        return Ordering::Less;
                    }
                    if left_non_primary.secondary() == NO_CE_SECONDARY {
                        break;
                    }
                }
            }
        }

        if let Some(tertiary_mask) = self.options.tertiary_mask() {
            let mut any_quaternaries = AnyQuaternaryAccumulator::new();
            let mut left_iter = left_ces.iter();
            let mut right_iter = right_ces.iter();
            loop {
                let mut left_non_primary;
                let mut left_tertiary;
                loop {
                    left_non_primary = left_iter.next().unwrap_or_default().non_primary();
                    any_quaternaries.accumulate(left_non_primary);
                    debug_assert!(
                        left_non_primary.tertiary() != 0 || left_non_primary.case_quaternary() == 0
                    );
                    left_tertiary = left_non_primary.tertiary_case_quarternary(tertiary_mask);
                    if left_tertiary != 0 {
                        break;
                    }
                }

                let mut right_non_primary;
                let mut right_tertiary;
                loop {
                    right_non_primary = right_iter.next().unwrap_or_default().non_primary();
                    any_quaternaries.accumulate(right_non_primary);
                    debug_assert!(
                        right_non_primary.tertiary() != 0
                            || right_non_primary.case_quaternary() == 0
                    );
                    right_tertiary = right_non_primary.tertiary_case_quarternary(tertiary_mask);
                    if right_tertiary != 0 {
                        break;
                    }
                }

                if left_tertiary != right_tertiary {
                    if self.options.upper_first() {
                        // Pass through NO_CE and keep real tertiary weights larger than that.
                        // Do not change the artificial uppercase weight of a tertiary CE (0.0.ut),
                        // to keep tertiary CEs well-formed.
                        // Their case+tertiary weights must be greater than those of
                        // primary and secondary CEs.
                        // Magic numbers from ICU4C.
                        if left_tertiary > NO_CE_TERTIARY {
                            if left_non_primary.secondary() != 0 {
                                left_tertiary ^= 0xC000;
                            } else {
                                left_tertiary += 0x4000;
                            }
                        }
                        if right_tertiary > NO_CE_TERTIARY {
                            if right_non_primary.secondary() != 0 {
                                right_tertiary ^= 0xC000;
                            } else {
                                right_tertiary += 0x4000;
                            }
                        }
                    }
                    if left_tertiary < right_tertiary {
                        return Ordering::Less;
                    }
                    return Ordering::Greater;
                }

                if left_tertiary == NO_CE_TERTIARY {
                    break;
                }
            }
            if !any_variable && !any_quaternaries.has_quaternary() {
                return Ordering::Equal;
            }
        } else {
            return Ordering::Equal;
        }

        if self.options.strength() <= Strength::Tertiary {
            return Ordering::Equal;
        }

        let mut left_iter = left_ces.iter();
        let mut right_iter = right_ces.iter();
        loop {
            let mut left_quaternary;
            loop {
                let ce = left_iter.next().unwrap_or_default();
                if ce.tertiary_ignorable() {
                    left_quaternary = ce.primary();
                } else {
                    left_quaternary = ce.quaternary();
                }
                if left_quaternary != 0 {
                    break;
                }
            }
            let mut right_quaternary;
            loop {
                let ce = right_iter.next().unwrap_or_default();
                if ce.tertiary_ignorable() {
                    right_quaternary = ce.primary();
                } else {
                    right_quaternary = ce.quaternary();
                }
                if right_quaternary != 0 {
                    break;
                }
            }
            if left_quaternary != right_quaternary {
                if let Some(reordering) = &self.reordering {
                    left_quaternary = reordering.get().reorder(left_quaternary);
                    right_quaternary = reordering.get().reorder(right_quaternary);
                }
                if left_quaternary < right_quaternary {
                    return Ordering::Less;
                }
                return Ordering::Greater;
            }
            if left_quaternary == NO_CE_PRIMARY {
                break;
            }
        }

        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atoi::FromRadix16;
    use icu_locid::langid;

    type StackString = arraystring::ArrayString<arraystring::typenum::U32>;

    /// Parse a string of space-separated hexadecimal code points (ending in end of input or semicolon)
    fn parse_hex(mut hexes: &[u8]) -> Option<StackString> {
        let mut buf = StackString::new();
        loop {
            let (scalar, mut offset) = u32::from_radix_16(hexes);
            if let Some(c) = core::char::from_u32(scalar) {
                buf.try_push(c).unwrap();
            } else {
                return None;
            }
            if offset == hexes.len() {
                return Some(buf);
            }
            match hexes[offset] {
                b';' => {
                    return Some(buf);
                }
                b' ' => {
                    offset += 1;
                }
                _ => {
                    panic!("Bad format: Garbage");
                }
            }
            hexes = &hexes[offset..];
        }
    }

    #[test]
    fn test_parse_hex() {
        assert_eq!(
            &parse_hex(b"1F926 1F3FC 200D 2642 FE0F").unwrap(),
            "\u{1F926}\u{1F3FC}\u{200D}\u{2642}\u{FE0F}"
        );
        assert_eq!(
            &parse_hex(b"1F926 1F3FC 200D 2642 FE0F; whatever").unwrap(),
            "\u{1F926}\u{1F3FC}\u{200D}\u{2642}\u{FE0F}"
        );
    }

    fn check_expectations(
        collator: &Collator,
        left: &[&str],
        right: &[&str],
        expectations: &[Ordering],
    ) {
        let mut left_iter = left.iter();
        let mut right_iter = right.iter();
        let mut expect_iter = expectations.iter();
        while let (Some(left_str), Some(right_str), Some(expectation)) =
            (left_iter.next(), right_iter.next(), expect_iter.next())
        {
            assert_eq!(collator.compare(left_str, right_str), *expectation);
        }
    }

    #[test]
    fn test_basic() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("ac", "äb"), Ordering::Greater);
    }

    #[test]
    fn test_implicit_unihan() {
        // Adapted from `CollationTest::TestImplicits()` in collationtest.cpp of ICU4C.
        // The radical-stroke order of the characters tested agrees with their code point
        // order.
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("\u{4E00}", "\u{4E00}"), Ordering::Equal);
        assert_eq!(collator.compare("\u{4E00}", "\u{4E01}"), Ordering::Less);
        assert_eq!(collator.compare("\u{4E01}", "\u{4E00}"), Ordering::Greater);

        assert_eq!(collator.compare("\u{4E18}", "\u{4E42}"), Ordering::Less);
        assert_eq!(collator.compare("\u{4E94}", "\u{50F6}"), Ordering::Less);
    }

    #[test]
    fn test_currency() {
        // Adapted from `CollationCurrencyTest::currencyTest` in currcoll.cpp of ICU4C.
        // All the currency symbols, in collation order.
        let currencies = "\u{00A4}\u{00A2}\u{FFE0}\u{0024}\u{FF04}\u{FE69}\u{00A3}\u{FFE1}\u{00A5}\u{FFE5}\u{09F2}\u{09F3}\u{0E3F}\u{17DB}\u{20A0}\u{20A1}\u{20A2}\u{20A3}\u{20A4}\u{20A5}\u{20A6}\u{20A7}\u{20A9}\u{FFE6}\u{20AA}\u{20AB}\u{20AC}\u{20AD}\u{20AE}\u{20AF}";
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        // Iterating as chars and re-encoding due to
        // https://github.com/rust-lang/rust/issues/83871 being nightly-only. :-(
        let mut lower_buf = [0u8; 4];
        let mut higher_buf = [0u8; 4];
        let mut chars = currencies.chars();
        while let Some(lower) = chars.next() {
            let mut tail = chars.clone();
            while let Some(higher) = tail.next() {
                let lower_str = lower.encode_utf8(&mut lower_buf);
                let higher_str = higher.encode_utf8(&mut higher_buf);
                assert_eq!(collator.compare(lower_str, higher_str), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_de() {
        // Adapted from `CollationGermanTest` in decoll.cpp of ICU4C.
        let left = [
            "\u{47}\u{72}\u{00F6}\u{00DF}\u{65}",
            "\u{61}\u{62}\u{63}",
            "\u{54}\u{00F6}\u{6e}\u{65}",
            "\u{54}\u{00F6}\u{6e}\u{65}",
            "\u{54}\u{00F6}\u{6e}\u{65}",
            "\u{61}\u{0308}\u{62}\u{63}",
            "\u{00E4}\u{62}\u{63}",
            "\u{00E4}\u{62}\u{63}",
            "\u{53}\u{74}\u{72}\u{61}\u{00DF}\u{65}",
            "\u{65}\u{66}\u{67}",
            "\u{00E4}\u{62}\u{63}",
            "\u{53}\u{74}\u{72}\u{61}\u{00DF}\u{65}",
        ];

        let right = [
            "\u{47}\u{72}\u{6f}\u{73}\u{73}\u{69}\u{73}\u{74}",
            "\u{61}\u{0308}\u{62}\u{63}",
            "\u{54}\u{6f}\u{6e}",
            "\u{54}\u{6f}\u{64}",
            "\u{54}\u{6f}\u{66}\u{75}",
            "\u{41}\u{0308}\u{62}\u{63}",
            "\u{61}\u{0308}\u{62}\u{63}",
            "\u{61}\u{65}\u{62}\u{63}",
            "\u{53}\u{74}\u{72}\u{61}\u{73}\u{73}\u{65}",
            "\u{65}\u{66}\u{67}",
            "\u{61}\u{65}\u{62}\u{63}",
            "\u{53}\u{74}\u{72}\u{61}\u{73}\u{73}\u{65}",
        ];

        let expect_primary = [
            Ordering::Less,
            Ordering::Equal,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Equal,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Equal,
        ];

        let expect_tertiary = [
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Greater,
        ];

        //        let locale: Locale = langid!("de").into();
        let locale: Locale = Locale::default(); // German uses the root collation

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Primary));

        let data_provider = icu_testdata::get_provider();
        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expect_primary);
        }

        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expect_tertiary);
        }
    }

    #[test]
    fn test_en() {
        // Adapted from encoll.cpp in ICU4C
        let left = [
            "\u{0061}\u{0062}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{002D}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0020}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{002D}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0048}\u{0065}\u{006C}\u{006C}\u{006F}",
            "\u{0041}\u{0042}\u{0043}", 
            "\u{0061}\u{0062}\u{0063}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{002D}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{002D}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0070}\u{00EA}\u{0063}\u{0068}\u{0065}",                                            
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{00E9}",
            "\u{00C4}\u{0042}\u{0308}\u{0043}\u{0308}",
            "\u{0061}\u{0308}\u{0062}\u{0063}",
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{0065}\u{0072}",
            "\u{0072}\u{006F}\u{006C}\u{0065}\u{0073}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0041}",
            "\u{0041}",
            "\u{0061}\u{0062}",                                                                
            "\u{0074}\u{0063}\u{006F}\u{006D}\u{0070}\u{0061}\u{0072}\u{0065}\u{0070}\u{006C}\u{0061}\u{0069}\u{006E}",
            "\u{0061}\u{0062}", 
            "\u{0061}\u{0023}\u{0062}",
            "\u{0061}\u{0023}\u{0062}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0041}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{0061}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{0061}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00E6}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00E4}\u{0062}\u{0063}\u{0064}\u{0061}",                                            
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0063}\u{0048}\u{0063}",
            "\u{0061}\u{0308}\u{0062}\u{0063}",
            "\u{0074}\u{0068}\u{0069}\u{0302}\u{0073}",
            "\u{0070}\u{00EA}\u{0063}\u{0068}\u{0065}",
            "\u{0061}\u{0062}\u{0063}",                                                         
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{00E6}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{00E6}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",               
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{00E9}"
        ]; // 49

        let right = [
            "\u{0061}\u{0062}\u{0063}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{002D}\u{0062}\u{0069}\u{0072}\u{0064}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}",
            "\u{0068}\u{0065}\u{006C}\u{006C}\u{006F}",
            "\u{0041}\u{0042}\u{0043}",
            "\u{0041}\u{0042}\u{0043}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0062}\u{0069}\u{0072}\u{0064}\u{0073}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0062}\u{0069}\u{0072}\u{0064}\u{0073}",
            "\u{0062}\u{006C}\u{0061}\u{0063}\u{006B}\u{0062}\u{0069}\u{0072}\u{0064}",                             
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{00E9}",
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{0065}\u{0072}",
            "\u{00C4}\u{0042}\u{0308}\u{0043}\u{0308}",
            "\u{0041}\u{0308}\u{0062}\u{0063}",
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{0065}",
            "\u{0072}\u{006F}\u{0302}\u{006C}\u{0065}",
            "\u{0041}\u{00E1}\u{0063}\u{0064}",
            "\u{0041}\u{00E1}\u{0063}\u{0064}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",                                                             
            "\u{0054}\u{0043}\u{006F}\u{006D}\u{0070}\u{0061}\u{0072}\u{0065}\u{0050}\u{006C}\u{0061}\u{0069}\u{006E}",
            "\u{0061}\u{0042}\u{0063}",
            "\u{0061}\u{0023}\u{0042}",
            "\u{0061}\u{0026}\u{0062}",
            "\u{0061}\u{0023}\u{0063}",
            "\u{0061}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00C4}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00E4}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00C4}\u{0062}\u{0063}\u{0064}\u{0061}",
            "\u{00C4}\u{0062}\u{0063}\u{0064}\u{0061}",                                             
            "\u{0061}\u{0062}\u{0023}\u{0063}",
            "\u{0061}\u{0062}\u{0063}",
            "\u{0061}\u{0062}\u{003D}\u{0063}",
            "\u{0061}\u{0062}\u{0064}",
            "\u{00E4}\u{0062}\u{0063}",
            "\u{0061}\u{0043}\u{0048}\u{0063}",
            "\u{00E4}\u{0062}\u{0063}",
            "\u{0074}\u{0068}\u{00EE}\u{0073}",
            "\u{0070}\u{00E9}\u{0063}\u{0068}\u{00E9}",
            "\u{0061}\u{0042}\u{0043}",                                                          
            "\u{0061}\u{0062}\u{0064}",
            "\u{00E4}\u{0062}\u{0063}",
            "\u{0061}\u{00C6}\u{0063}",
            "\u{0061}\u{0042}\u{0064}",
            "\u{00E4}\u{0062}\u{0063}",
            "\u{0061}\u{00C6}\u{0063}",
            "\u{0061}\u{0042}\u{0064}",
            "\u{00E4}\u{0062}\u{0063}",          
            "\u{0070}\u{00EA}\u{0063}\u{0068}\u{0065}"
        ]; // 49

        let expectations = [
            Ordering::Less,
            Ordering::Less, /*Ordering::Greater,*/
            Ordering::Less,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less, /*Ordering::Greater,*/
            /* 10 */
            Ordering::Greater,
            Ordering::Less,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less, /* 20 */
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Greater,
            Ordering::Greater,
            /* Test Tertiary  > 26 */
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less, /* 30 */
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            /* test identical > 36 */
            Ordering::Equal,
            Ordering::Equal,
            /* test primary > 38 */
            Ordering::Equal,
            Ordering::Equal, /* 40 */
            Ordering::Less,
            Ordering::Equal,
            Ordering::Equal,
            /* test secondary > 43 */
            Ordering::Less,
            Ordering::Less,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less, // 49
        ];

        //        let locale: Locale = langid!("en").into();
        let locale: Locale = Locale::default(); // English uses the root collation
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left[..38], &right[..38], &expectations[..38]);
        }

        options.set_strength(Some(Strength::Primary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(
                &collator,
                &left[38..43],
                &right[38..43],
                &expectations[38..43],
            );
        }

        options.set_strength(Some(Strength::Secondary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left[43..], &right[43..], &expectations[43..]);
        }
    }

    #[test]
    fn test_en_bugs() {
        // Adapted from encoll.cpp in ICU4C
        let bugs = [
            "\u{61}",
            "\u{41}",
            "\u{65}",
            "\u{45}",
            "\u{00e9}",
            "\u{00e8}",
            "\u{00ea}",
            "\u{00eb}",
            "\u{65}\u{61}",
            "\u{78}",
        ];
        //        let locale: Locale = langid!("en").into();
        let locale: Locale = Locale::default(); // English uses the root collation
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            let mut outer = bugs.iter();
            while let Some(left) = outer.next() {
                let mut inner = outer.clone();
                while let Some(right) = inner.next() {
                    assert_eq!(collator.compare(left, right), Ordering::Less);
                }
            }
        }
    }

    #[test]
    fn test_ja_tertiary() {
        // Adapted from `CollationKanaTest::TestTertiary` in jacoll.cpp in ICU4C
        let left = [
            "\u{FF9E}",
            "\u{3042}",
            "\u{30A2}",
            "\u{3042}\u{3042}",
            "\u{30A2}\u{30FC}",
            "\u{30A2}\u{30FC}\u{30C8}",
        ];
        let right = [
            "\u{FF9F}",
            "\u{30A2}",
            "\u{3042}\u{3042}",
            "\u{30A2}\u{30FC}",
            "\u{30A2}\u{30FC}\u{30C8}",
            "\u{3042}\u{3042}\u{3068}",
        ];
        let expectations = [
            Ordering::Less,
            Ordering::Equal, // Katakanas and Hiraganas are equal on tertiary level
            Ordering::Less,
            Ordering::Greater, // Prolonged sound mark sorts BEFORE equivalent vowel
            Ordering::Less,
            Ordering::Less, // Prolonged sound mark sorts BEFORE equivalent vowel
        ];
        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));
        options.set_case_level(Some(true));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_ja_base() {
        // Adapted from `CollationKanaTest::TestBase` in jacoll.cpp of ICU4C.
        let cases = [
            "\u{30AB}",
            "\u{30AB}\u{30AD}",
            "\u{30AD}",
            "\u{30AD}\u{30AD}",
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Primary));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(collator.compare(lower, higher), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_ja_plain_dakuten_handakuten() {
        // Adapted from `CollationKanaTest::TestPlainDakutenHandakuten` in jacoll.cpp of ICU4C.
        let cases = [
            "\u{30CF}\u{30AB}",
            "\u{30D0}\u{30AB}",
            "\u{30CF}\u{30AD}",
            "\u{30D0}\u{30AD}",
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Secondary));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(collator.compare(lower, higher), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_ja_small_large() {
        // Adapted from `CollationKanaTest::TestSmallLarge` in jacoll.cpp of ICU4C.
        let cases = [
            "\u{30C3}\u{30CF}",
            "\u{30C4}\u{30CF}",
            "\u{30C3}\u{30D0}",
            "\u{30C4}\u{30D0}",
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));
        options.set_case_level(Some(true));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(collator.compare(lower, higher), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_ja_hiragana_katakana() {
        // Adapted from `CollationKanaTest::TestKatakanaHiragana` in jacoll.cpp of ICU4C.
        let cases = [
            "\u{3042}\u{30C3}",
            "\u{30A2}\u{30C3}",
            "\u{3042}\u{30C4}",
            "\u{30A2}\u{30C4}",
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));
        options.set_case_level(Some(true));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(collator.compare(lower, higher), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_ja_hiragana_katakana_utf16() {
        // Adapted from `CollationKanaTest::TestKatakanaHiragana` in jacoll.cpp of ICU4C.
        let cases = [
            &[0x3042u16, 0x30C3u16],
            &[0x30A2u16, 0x30C3u16],
            &[0x3042u16, 0x30C4u16],
            &[0x30A2u16, 0x30C4u16],
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));
        options.set_case_level(Some(true));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(
                    collator.compare_utf16(&lower[..], &higher[..]),
                    Ordering::Less
                );
            }
        }
    }

    #[test]
    fn test_ja_chooon_kigoo() {
        // Adapted from `CollationKanaTest::TestChooonKigoo` in jacoll.cpp of ICU4C.
        let cases = [
            "\u{30AB}\u{30FC}\u{3042}",
            "\u{30AB}\u{30FC}\u{30A2}",
            "\u{30AB}\u{30A4}\u{3042}",
            "\u{30AB}\u{30A4}\u{30A2}",
            "\u{30AD}\u{30FC}\u{3042}", /* Prolonged sound mark sorts BEFORE equivalent vowel (ICU 2.0)*/
            "\u{30AD}\u{30FC}\u{30A2}", /* Prolonged sound mark sorts BEFORE equivalent vowel (ICU 2.0)*/
            "\u{30AD}\u{30A4}\u{3042}",
            "\u{30AD}\u{30A4}\u{30A2}",
        ];

        let locale: Locale = langid!("ja").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));
        options.set_case_level(Some(true));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut case_iter = cases.iter();
        while let Some(lower) = case_iter.next() {
            let mut tail = case_iter.clone();
            while let Some(higher) = tail.next() {
                assert_eq!(collator.compare(lower, higher), Ordering::Less);
            }
        }
    }

    #[test]
    fn test_fi() {
        // Adapted from ficoll.cpp in ICU4C
        // Testing that w and v behave as in the root collation is for checking
        // that the sorting collation doesn't exhibit the behavior of the search
        // collation, which (somewhat questionably) treats w and v as primary-equal.
        let left = [
            "wat",
            "vat",
            "a\u{FC}beck",
            "L\u{E5}vi",
            // ICU4C has a duplicate of the first case below.
            // The duplicate is omitted here.
            // Instead, the subsequent tests are added for ICU4X.
            "\u{E4}",
            "a\u{0308}",
        ];
        let right = [
            "vat",
            "way",
            "axbeck",
            "L\u{E4}we",
            // ICU4C has a duplicate of the first case below.
            // The duplicate is omitted here.
            // Instead, the subsequent tests are added for ICU4X.
            "o",
            "\u{E4}",
        ];
        let expectations = [
            Ordering::Greater,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Equal,
        ];
        let locale: Locale = langid!("fi").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }

        options.set_strength(Some(Strength::Primary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_sv() {
        // This is the same as test_fi. The purpose of this copy is to test that
        // Swedish defaults to "reformed", which behaves like Finnish "standard",
        // and not to "standard", which behaves like Finnash "traditional".

        // Adapted from ficoll.cpp in ICU4C
        // Testing that w and v behave as in the root collation is for checking
        // that the sorting collation doesn't exhibit the behavior of the search
        // collation, which (somewhat questionably) treats w and v as primary-equal.
        let left = [
            "wat",
            "vat",
            "a\u{FC}beck",
            "L\u{E5}vi",
            // ICU4C has a duplicate of the first case below.
            // The duplicate is omitted here.
            // Instead, the subsequent tests are added for ICU4X.
            "\u{E4}",
            "a\u{0308}",
        ];
        let right = [
            "vat",
            "way",
            "axbeck",
            "L\u{E4}we",
            // ICU4C has a duplicate of the first case below.
            // The duplicate is omitted here.
            // Instead, the subsequent tests are added for ICU4X.
            "o",
            "\u{E4}",
        ];
        let expectations = [
            Ordering::Greater,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Equal,
        ];
        let locale: Locale = langid!("sv").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }

        options.set_strength(Some(Strength::Primary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    // TODO: This test should eventually test fallback
    // TODO: Test Swedish and Chinese also, since they have unusual
    // variant defaults. (But are currently not part of the test data.)
    #[test]
    fn test_region_fallback() {
        // There's no explicit fi-FI data.
        let locale: Locale = "fi-u-co-standard".parse().unwrap();

        // let locale: Locale = langid!("fi-FI").into();

        let data_provider = icu_testdata::get_provider();

        let collator: Collator =
            Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
        assert_eq!(collator.compare("ä", "z"), Ordering::Greater);
    }

    #[test]
    fn test_reordering() {
        let locale: Locale = langid!("bn").into();
        let data_provider = icu_testdata::get_provider();

        {
            let collator: Collator =
                Collator::try_new(Locale::default(), &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("অ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ऄ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("অ", "ऄ"), Ordering::Greater);
        }

        {
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("অ", "a"), Ordering::Less);
            assert_eq!(collator.compare("ऄ", "a"), Ordering::Less);
            assert_eq!(collator.compare("অ", "ऄ"), Ordering::Less);
        }
    }

    #[test]
    fn test_zh() {
        let data_provider = icu_testdata::get_provider();

        // Note: ㄅ is Bopomofo.

        {
            let collator: Collator =
                Collator::try_new(Locale::default(), &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Greater);
            assert_eq!(collator.compare("佰", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Greater);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Greater);
            assert_eq!(collator.compare("不", "把"), Ordering::Less);
        }
        {
            let locale: Locale = langid!("zh").into(); // Defaults to -u-co-pinyin
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Less);
            assert_eq!(collator.compare("佰", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Less);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Greater);
        }
        {
            let locale: Locale = "zh-u-co-pinyin".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Less);
            assert_eq!(collator.compare("佰", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Less);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Greater);
        }
        {
            let locale: Locale = "zh-u-co-gb2312".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Greater);
            assert_eq!(collator.compare("佰", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Less);
            // In GB2312 proper, Bopomofo comes before Han, but the
            // collation leaves Bopomofo unreordered, so it comes after.
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Greater);
        }
        {
            let locale: Locale = "zh-u-co-stroke".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Less);
            assert_eq!(collator.compare("佰", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Less);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Less);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Less);
        }
        {
            let locale: Locale = "zh-u-co-zhuyin".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Less);
            assert_eq!(collator.compare("佰", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Less);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Greater);
        }
        {
            let locale: Locale = "zh-u-co-unihan".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Less);
            assert_eq!(collator.compare("佰", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Less);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Less);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Greater);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Less);
        }
        {
            let locale: Locale = "zh-u-co-big5han".parse().unwrap();
            let collator: Collator =
                Collator::try_new(locale, &data_provider, CollatorOptions::new()).unwrap();
            assert_eq!(collator.compare("艾", "a"), Ordering::Greater);
            assert_eq!(collator.compare("佰", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "a"), Ordering::Greater);
            assert_eq!(collator.compare("ㄅ", "ж"), Ordering::Less);
            assert_eq!(collator.compare("艾", "佰"), Ordering::Less);
            assert_eq!(collator.compare("艾", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("佰", "ㄅ"), Ordering::Less);
            assert_eq!(collator.compare("不", "把"), Ordering::Less);
        }
        // TODO: Test script and region aliases
    }

    // TODO: frcoll requires support for fr-CA

    // TODO: Write a test for Bangla

    #[test]
    fn test_es_tertiary() {
        // Adapted from `CollationSpanishTest::TestTertiary` in escoll.cpp in ICU4C
        let left = [
            "\u{61}\u{6c}\u{69}\u{61}\u{73}",
            "\u{45}\u{6c}\u{6c}\u{69}\u{6f}\u{74}",
            "\u{48}\u{65}\u{6c}\u{6c}\u{6f}",
            "\u{61}\u{63}\u{48}\u{63}",
            "\u{61}\u{63}\u{63}",
        ];
        let right = [
            "\u{61}\u{6c}\u{6c}\u{69}\u{61}\u{73}",
            "\u{45}\u{6d}\u{69}\u{6f}\u{74}",
            "\u{68}\u{65}\u{6c}\u{6c}\u{4f}",
            "\u{61}\u{43}\u{48}\u{63}",
            "\u{61}\u{43}\u{48}\u{63}",
        ];
        let expectations = [
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Less,
        ];
        let locale: Locale = langid!("es").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_es_primary() {
        // Adapted from `CollationSpanishTest::TestPrimary` in escoll.cpp in ICU4C
        let left = [
            "\u{61}\u{6c}\u{69}\u{61}\u{73}",
            "\u{61}\u{63}\u{48}\u{63}",
            "\u{61}\u{63}\u{63}",
            "\u{48}\u{65}\u{6c}\u{6c}\u{6f}",
        ];
        let right = [
            "\u{61}\u{6c}\u{6c}\u{69}\u{61}\u{73}",
            "\u{61}\u{43}\u{48}\u{63}",
            "\u{61}\u{43}\u{48}\u{63}",
            "\u{68}\u{65}\u{6c}\u{6c}\u{4f}",
        ];
        let expectations = [
            Ordering::Less,
            Ordering::Equal,
            Ordering::Less,
            Ordering::Equal,
        ];
        let locale: Locale = langid!("es").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Primary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_el_secondary() {
        // Adapted from `CollationRegressionTest::Test4095316` in regcoll.cpp of ICU4C.
        let locale: Locale = Locale::default(); // Greek uses the root collation
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Secondary));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("\u{03D4}", "\u{03AB}"), Ordering::Equal);
    }

    #[test]
    fn test_th_dictionary() {
        // Adapted from `CollationThaiTest::TestDictionary` of thcoll.cpp in ICU4C.
        let dict = include_str!("riwords.txt")
            .strip_prefix('\u{FEFF}')
            .unwrap();
        let locale: Locale = langid!("th").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        let collator: Collator =
            Collator::try_new(locale.clone(), &data_provider, options).unwrap();
        let mut lines = dict.lines();
        let mut prev = loop {
            if let Some(line) = lines.next() {
                if line.starts_with('#') {
                    continue;
                }
                break line;
            } else {
                panic!("Malformed dictionary");
            }
        };

        while let Some(line) = lines.next() {
            assert_eq!(collator.compare(prev, line), Ordering::Less);
            prev = line;
        }
    }

    #[test]
    fn test_th_corner_cases() {
        // Adapted from `CollationThaiTest::TestCornerCases` in thcoll.cpp in ICU4C
        let left = [
            // Shorter words precede longer
            "\u{0E01}",
            // Tone marks are considered after letters (i.e. are primary ignorable)
            "\u{0E01}\u{0E32}",
            // ditto for other over-marks
            "\u{0E01}\u{0E32}",
            // commonly used mark-in-context order.
            // In effect, marks are sorted after each syllable.
            "\u{0E01}\u{0E32}\u{0E01}\u{0E49}\u{0E32}",
            // Hyphens and other punctuation follow whitespace but come before letters
            //            "\u{0E01}\u{0E32}",
            "\u{0E01}\u{0E32}-",
            // Doubler follows an identical word without the doubler
            // "\u{0E01}\u{0E32}",
            "\u{0E01}\u{0E32}\u{0E46}",
            // \u{0E45} after either \u{0E24} or \{u0E26} is treated as a single
            // combining character, similar to "c < ch" in traditional spanish.
            "\u{0E24}\u{0E29}\u{0E35}",
            "\u{0E26}\u{0E29}\u{0E35}",
            // Vowels reorder, should compare \u{0E2D} and \u{0E34}
            // XXX Should the middle code point differ between the two strings?
            "\u{0E40}\u{0E01}\u{0E2D}",
            // Tones are compared after the rest of the word (e.g. primary ignorable)
            "\u{0E01}\u{0E32}\u{0E01}\u{0E48}\u{0E32}",
            // Periods are ignored entirely
            "\u{0E01}.\u{0E01}.",
        ];
        let right = [
            "\u{0E01}\u{0E01}",
            "\u{0E01}\u{0E49}\u{0E32}",
            "\u{0E01}\u{0E32}\u{0E4C}",
            "\u{0E01}\u{0E48}\u{0E32}\u{0E01}\u{0E49}\u{0E32}",
            //            "\u{0E01}\u{0E32}-",
            "\u{0E01}\u{0E32}\u{0E01}\u{0E32}",
            // "\u{0E01}\u{0E32}\u{0E46}",
            "\u{0E01}\u{0E32}\u{0E01}\u{0E32}",
            "\u{0E24}\u{0E45}\u{0E29}\u{0E35}",
            "\u{0E26}\u{0E45}\u{0E29}\u{0E35}",
            "\u{0E40}\u{0E01}\u{0E34}",
            "\u{0E01}\u{0E49}\u{0E32}\u{0E01}\u{0E32}",
            "\u{0E01}\u{0E32}",
        ];
        let expectations = [
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            //            Ordering::Equal,
            Ordering::Less,
            // Ordering::Equal,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
        ];
        let locale: Locale = langid!("th").into();
        let data_provider = icu_testdata::get_provider();
        {
            // XXX TODO: Check why the commented-out cases fail
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, CollatorOptions::new()).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_th_reordering() {
        // Adapted from `CollationThaiTest::TestReordering` in thcoll.cpp in ICU4C
        let left = [
            // composition
            "\u{0E41}c\u{0301}",
            // supplementaries
            // XXX Why does this fail?
            // "\u{0E41}\u{1D7CE}",
            // supplementary composition decomps to supplementary
            "\u{0E41}\u{1D15F}",
            // supplementary composition decomps to BMP
            "\u{0E41}\u{2F802}", // omit bacward iteration tests
                                 // contraction bug
                                 // "\u{0E24}\u{0E41}",
                                 // TODO: Support contracting starters, then add more here
        ];
        let right = [
            "\u{0E41}\u{0107}",
            // "\u{0E41}\u{1D7CF}",
            "\u{0E41}\u{1D158}\u{1D165}",
            "\u{0E41}\u{4E41}", // "\u{0E41}\u{0E24}",
        ];
        let expectations = [
            Ordering::Equal,
            // Ordering::Less,
            Ordering::Equal,
            Ordering::Equal,
            // Ordering::Equal,
        ];
        let locale: Locale = langid!("th").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Secondary));

        {
            // XXX TODO: Thai requires shifted alternate handling
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_tr_tertiary() {
        // Adapted from `CollationTurkishTest::TestTertiary` in trcoll.cpp in ICU4C
        let left = [
            "\u{73}\u{0327}",
            "\u{76}\u{00E4}\u{74}",
            "\u{6f}\u{6c}\u{64}",
            "\u{00FC}\u{6f}\u{69}\u{64}",
            "\u{68}\u{011E}\u{61}\u{6c}\u{74}",
            "\u{73}\u{74}\u{72}\u{65}\u{73}\u{015E}",
            "\u{76}\u{6f}\u{0131}\u{64}",
            "\u{69}\u{64}\u{65}\u{61}",
        ];
        let right = [
            "\u{75}\u{0308}",
            "\u{76}\u{62}\u{74}",
            "\u{00D6}\u{61}\u{79}",
            "\u{76}\u{6f}\u{69}\u{64}",
            "\u{68}\u{61}\u{6c}\u{74}",
            "\u{015E}\u{74}\u{72}\u{65}\u{015E}\u{73}",
            "\u{76}\u{6f}\u{69}\u{64}",
            "\u{49}\u{64}\u{65}\u{61}",
        ];
        let expectations = [
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
        ];
        let locale: Locale = langid!("tr").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_tr_primary() {
        // Adapted from `CollationTurkishTest::TestPrimary` in trcoll.cpp in ICU4C
        let left = [
            "\u{00FC}\u{6f}\u{69}\u{64}",
            "\u{76}\u{6f}\u{0131}\u{64}",
            "\u{69}\u{64}\u{65}\u{61}",
        ];
        let right = [
            "\u{76}\u{6f}\u{69}\u{64}",
            "\u{76}\u{6f}\u{69}\u{64}",
            "\u{49}\u{64}\u{65}\u{61}",
        ];
        let expectations = [Ordering::Less, Ordering::Less, Ordering::Greater];
        let locale: Locale = langid!("tr").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_lt_tertiary() {
        let left = [
            "a\u{0307}\u{0300}a",
            "a\u{0307}\u{0301}a",
            "a\u{0307}\u{0302}a",
            "a\u{0307}\u{0303}a",
            "ž",
        ];
        let right = [
            "a\u{0300}a",
            "a\u{0301}a",
            "a\u{0302}a",
            "a\u{0303}a",
            "z\u{033F}",
        ];
        let expectations = [
            Ordering::Equal,
            Ordering::Equal,
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Greater,
        ];
        let locale: Locale = langid!("lt").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_lt_primary() {
        let left = ["ž"];
        let right = ["z"];
        let expectations = [Ordering::Greater];
        let locale: Locale = langid!("lt").into();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Primary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_basics() {
        // Adapted from `CollationAPITest::TestProperty` in apicoll.cpp in ICU4C
        let left = ["ab", "ab", "blackbird", "black bird", "Hello", "", "abä"];
        let right = ["abc", "AB", "black-bird", "black-bird", "hello", "", "abß"];
        let expectations = [
            Ordering::Less,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Less,
            Ordering::Greater,
            Ordering::Equal,
        ];
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Tertiary));

        {
            let collator: Collator =
                Collator::try_new(Locale::default(), &data_provider, options).unwrap();
            check_expectations(&collator, &left, &right, &expectations);
        }
    }

    #[test]
    fn test_numeric_off() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_numeric(Some(false));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("a10b", "a2b"), Ordering::Less);
    }

    #[test]
    fn test_numeric_on() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_numeric(Some(true));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("a10b", "a2b"), Ordering::Greater);
    }

    #[test]
    fn test_numeric_long() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_numeric(Some(true));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        let mut left = String::new();
        let mut right = String::new();
        // We'll make left larger than right numerically. However, first, let's use
        // a leading zero to make left look less than right non-numerically.
        left.push('0');
        left.push('1');
        right.push('1');
        // Now, let's make sure we end up with segments longer than
        // 254 digits
        for _ in 0..256 {
            left.push('2');
            right.push('2');
        }
        // Now the difference:
        left.push('4');
        right.push('3');
        // Again making segments long
        for _ in 0..256 {
            left.push('5');
            right.push('5');
        }
        // And some trailing zeros
        for _ in 0..7 {
            left.push('0');
            right.push('0');
        }
        assert_eq!(collator.compare(&left, &right), Ordering::Greater);
    }

    #[test]
    fn test_numeric_after() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_numeric(Some(true));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(collator.compare("0001000b", "1000a"), Ordering::Greater);
    }

    #[test]
    fn test_unpaired_surrogates() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        assert_eq!(
            collator.compare_utf16(&[0xD801u16], &[0xD802u16]),
            Ordering::Equal
        );
    }

    #[test]
    fn test_backward_second_level() {
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Secondary));

        {
            let collator: Collator =
                Collator::try_new(Locale::default(), &data_provider, options).unwrap();

            let cases = ["cote", "coté", "côte", "côté"];
            let mut case_iter = cases.iter();
            while let Some(lower) = case_iter.next() {
                let mut tail = case_iter.clone();
                while let Some(higher) = tail.next() {
                    assert_eq!(collator.compare(lower, higher), Ordering::Less);
                }
            }
        }

        options.set_backward_second_level(Some(true));

        {
            let collator: Collator =
                Collator::try_new(Locale::default(), &data_provider, options).unwrap();

            let cases = ["cote", "côte", "coté", "côté"];
            let mut case_iter = cases.iter();
            while let Some(lower) = case_iter.next() {
                let mut tail = case_iter.clone();
                while let Some(higher) = tail.next() {
                    assert_eq!(collator.compare(lower, higher), Ordering::Less);
                }
            }
        }
    }

    #[test]
    fn test_cantillation() {
        let locale: Locale = Locale::default();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            assert_eq!(
                collator.compare(
                    "\u{05D3}\u{05D7}\u{05D9}\u{05AD}",
                    "\u{05D3}\u{05D7}\u{05D9}"
                ),
                Ordering::Equal
            );
        }

        options.set_strength(Some(Strength::Identical));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            assert_eq!(
                collator.compare(
                    "\u{05D3}\u{05D7}\u{05D9}\u{05AD}",
                    "\u{05D3}\u{05D7}\u{05D9}"
                ),
                Ordering::Greater
            );
        }
    }

    #[test]
    fn test_cantillation_utf8() {
        let locale: Locale = Locale::default();
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            assert_eq!(
                collator.compare_utf8(
                    "\u{05D3}\u{05D7}\u{05D9}\u{05AD}".as_bytes(),
                    "\u{05D3}\u{05D7}\u{05D9}".as_bytes()
                ),
                Ordering::Equal
            );
        }

        options.set_strength(Some(Strength::Identical));

        {
            let collator: Collator =
                Collator::try_new(locale.clone(), &data_provider, options).unwrap();
            assert_eq!(
                collator.compare(
                    "\u{05D3}\u{05D7}\u{05D9}\u{05AD}",
                    "\u{05D3}\u{05D7}\u{05D9}"
                ),
                Ordering::Greater
            );
        }
    }

    #[test]
    fn test_conformance_shifted() {
        // Adapted from `UCAConformanceTest::TestTableShifted` of ucaconf.cpp in ICU4C.
        let bugs = [];
        let dict = include_bytes!("CollationTest_CLDR_SHIFTED.txt");
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));
        options.set_alternate_handling(Some(AlternateHandling::Shifted));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        let mut lines = dict.split(|b| b == &b'\n');
        let mut prev = loop {
            if let Some(line) = lines.next() {
                if line.is_empty() {
                    continue;
                }
                if line.starts_with(&[b'#']) {
                    continue;
                }
                if let Some(parsed) = parse_hex(line) {
                    break parsed;
                }
            } else {
                panic!("Malformed dictionary");
            }
        };

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            if let Some(parsed) = parse_hex(line) {
                if !bugs.contains(&parsed.as_str()) {
                    if collator.compare(&prev, &parsed) == Ordering::Greater {
                        assert_eq!(&prev[..], &parsed[..]);
                    }
                }
                prev = parsed;
            }
        }
    }

    #[test]
    fn test_conformance_non_ignorable() {
        // Adapted from `UCAConformanceTest::TestTableNonIgnorable` of ucaconf.cpp in ICU4C.
        let bugs = [];
        let dict = include_bytes!("CollationTest_CLDR_NON_IGNORABLE.txt");
        let data_provider = icu_testdata::get_provider();

        let mut options = CollatorOptions::new();
        options.set_strength(Some(Strength::Quaternary));
        options.set_alternate_handling(Some(AlternateHandling::NonIgnorable));

        let collator: Collator = Collator::try_new(Locale::default(), &data_provider, options).unwrap();
        let mut lines = dict.split(|b| b == &b'\n');
        let mut prev = loop {
            if let Some(line) = lines.next() {
                if line.is_empty() {
                    continue;
                }
                if line.starts_with(&[b'#']) {
                    continue;
                }
                if let Some(parsed) = parse_hex(line) {
                    break parsed;
                }
            } else {
                panic!("Malformed dictionary");
            }
        };

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            if let Some(parsed) = parse_hex(line) {
                if !bugs.contains(&parsed.as_str()) {
                    if collator.compare(&prev, &parsed) == Ordering::Greater {
                        assert_eq!(&prev[..], &parsed[..]);
                    }
                }
                prev = parsed;
            }
        }
    }
}

// TODO: Test languages that map to the root.
// The languages that map to root without script reordering are:
// ca (at least for now)
// de
// en
// fr
// ga
// id
// it
// lb
// ms
// nl
// pt
// sw
// xh
// zu
//
// A bit unclear: ff in Latin script?
//
// These are root plus script reordering:
// am
// be
// bg
// chr
// el
// ka
// lo
// mn
// ne
// ru

// TODO: Test imports. These aren't aliases but should get deduplicated
// in the provider:
// bo: dz-u-co-standard (draft: unconfirmed)
// bs: hr
// bs-Cyrl: sr
// fa-AF: ps
// sr-Latn: hr

// TODO: Test that nn and nb are aliases for no

// TODO: Consider testing ff-Adlm for supplementary-plane tailoring, including contractions

// TODO: Test Tibetan

// TODO: Test de-AT-u-co-phonebk vs de-DE-u-co-phonebk

// TODO: Test da defaulting to [caseFirst upper]
// TODO: Test fr-CA defaulting to backward second level
