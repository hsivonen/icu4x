// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

#![cfg_attr(not(any(test, feature = "std")), no_std)]

//! `icu_normalizer` is one of the [`ICU4X`] components.
//!
//! This API provides necessary functionality for normalizing text into Unicode
//! Normalization Forms.

extern crate alloc;

pub mod error;
pub mod provider;

use crate::error::NormalizerError;
use crate::provider::CanonicalDecompositionDataV1;
use crate::provider::CanonicalDecompositionDataV1Marker;
use alloc::string::String;
use alloc::vec::Vec;
use core::char::{decode_utf16, DecodeUtf16Error, REPLACEMENT_CHARACTER};
use icu_codepointtrie::CodePointTrie;
use icu_properties::provider::UnicodePropertyMapV1Marker;
use icu_properties::CanonicalCombiningClass;
use icu_provider::DataPayload;
use icu_provider::DataRequest;
use icu_provider::DynProvider;
use icu_provider::ResourceProvider;
use smallvec::SmallVec;
use utf8_iter::Utf8CharsEx;
use zerovec::ule::AsULE;

// These constants originate from page 143 of Unicode 14.0
const HANGUL_S_BASE: u32 = 0xAC00;
const HANGUL_L_BASE: u32 = 0x1100;
const HANGUL_V_BASE: u32 = 0x1161;
const HANGUL_T_BASE: u32 = 0x11A7;
const HANGUL_T_COUNT: u32 = 28;
const HANGUL_N_COUNT: u32 = 588;
const HANGUL_S_COUNT: u32 = 11172;

#[inline(always)]
fn in_inclusive_range(c: char, start: char, end: char) -> bool {
    u32::from(c).wrapping_sub(u32::from(start)) <= (u32::from(end) - u32::from(start))
}

// Hoisted to function, because the compiler doesn't like having
// to identical closures.
#[inline(always)]
fn utf16_error_to_replacement(r: Result<char, DecodeUtf16Error>) -> char {
    r.unwrap_or(REPLACEMENT_CHARACTER)
}

/// Pack a `char` and a `CanonicalCombiningClass` in
/// 32 bits. The latter is initialized to 0 upon
/// creation and can be set by calling `set_ccc`.
/// This type is intentionally non-`Copy` to get
/// compiler help in making sure that the class is
/// set on the instance on which it is intended to
/// be set and not on a temporary copy.
///
/// Note: As a micro-optimization, this struct is
/// distinct from the struct of the same name in
/// the collator. They have the opposite default
/// bit pattern for the high 8 bits.
#[derive(Debug)]
struct CharacterAndClass(u32);

impl CharacterAndClass {
    pub fn new(c: char) -> Self {
        // The combining class bits default to zero.
        CharacterAndClass(u32::from(c))
    }
    pub fn character(&self) -> char {
        unsafe { char::from_u32_unchecked(self.0 & 0xFFFFFF) }
    }
    pub fn ccc(&self) -> CanonicalCombiningClass {
        CanonicalCombiningClass((self.0 >> 24) as u8)
    }
    // XXX need better naming here.
    pub fn set_ccc(&mut self, ccc: &CodePointTrie<CanonicalCombiningClass>) {
        debug_assert_eq!(self.0 >> 24, 0, "This method has already been called!");
        self.0 |= (ccc.get(self.0).0 as u32) << 24;
    }
}

// This function exists as a borrow check helper.
#[inline(always)]
fn sort_slice_by_ccc<'data>(
    slice: &mut [CharacterAndClass],
    ccc: &CodePointTrie<'data, CanonicalCombiningClass>,
) {
    // We don't look up the canonical combining class for starters
    // of for single combining characters between starters. When
    // there's more than one combining character between starters,
    // we look up the canonical combining class for each character
    // exactly once.
    if slice.len() < 2 {
        return;
    }
    slice.iter_mut().for_each(|cc| cc.set_ccc(ccc));
    slice.sort_by_key(|cc| cc.ccc());
}

pub struct Decomposition<'data, I>
where
    I: Iterator<Item = char>,
{
    delegate: I,
    buffer: SmallVec<[CharacterAndClass; 10]>, // TODO Figure out good length
    pending_unnormalized_starter: Option<char>, // None at end of stream
    decompositions: &'data CanonicalDecompositionDataV1<'data>,
    ccc: &'data CodePointTrie<'data, CanonicalCombiningClass>,
}

impl<'data, I> Decomposition<'data, I>
where
    I: Iterator<Item = char>,
{
    pub fn new(
        delegate: I,
        decompositions: &'data CanonicalDecompositionDataV1,
        ccc: &'data CodePointTrie<'data, CanonicalCombiningClass>,
    ) -> Self {
        let mut ret = Decomposition::<I> {
            delegate: delegate,
            buffer: SmallVec::new(), // Normalized
            // Initialize with a placeholder starter in case
            // the real stream starts with a non-starter.
            pending_unnormalized_starter: Some('\u{FFFF}'),
            decompositions: decompositions,
            ccc: ccc,
        };
        let _ = ret.next(); // Remove the U+FFFF placeholder
        ret
    }
}

impl<'data, I> Iterator for Decomposition<'data, I>
where
    I: Iterator<Item = char>,
{
    type Item = char;

    fn next(&mut self) -> Option<char> {
        if !self.buffer.is_empty() {
            return Some(self.buffer.remove(0).character());
        }
        let c = if let Some(c) = self.pending_unnormalized_starter.take() {
            c
        } else {
            return None;
        };
        let (starter, combining_start) = {
            let hangul_offset = u32::from(c).wrapping_sub(HANGUL_S_BASE); // SIndex in the spec
            if hangul_offset >= HANGUL_S_COUNT {
                let decomposition = self.decompositions.trie.get(u32::from(c));
                if decomposition == 0 {
                    // The character is its own decomposition
                    (c, 0)
                } else {
                    let high = (decomposition >> 16) as u16;
                    let low = decomposition as u16;
                    if high != 0 && low != 0 {
                        // Decomposition into two BMP characters: starter and non-starter
                        let starter = core::char::from_u32(u32::from(high)).unwrap();
                        let combining = core::char::from_u32(u32::from(low)).unwrap();
                        self.buffer.push(CharacterAndClass::new(combining));
                        (starter, 0)
                    } else if high != 0 {
                        // Decomposition into one BMP character
                        let starter = core::char::from_u32(u32::from(high)).unwrap();
                        (starter, 0)
                    } else {
                        // Complex decomposition
                        let offset = usize::from(low & 0x7FF);
                        let len = usize::from(low >> 13);
                        if low & 0x1000 == 0 {
                            let (&first, tail) = &self.decompositions.scalars16.as_ule_slice()
                                [offset..offset + len]
                                .split_first()
                                .unwrap();
                            // Starter
                            let starter =
                                core::char::from_u32(u32::from(u16::from_unaligned(first)))
                                    .unwrap();
                            if low & 0x800 == 0 {
                                // All the rest are combining
                                for &ule in tail.iter() {
                                    self.buffer.push(CharacterAndClass::new(
                                        core::char::from_u32(u32::from(u16::from_unaligned(ule)))
                                            .unwrap(),
                                    ));
                                }
                                (starter, 0)
                            } else {
                                let mut i = 0;
                                let mut combining_start = 0;
                                let mut it = tail.iter();
                                while let Some(&ule) = it.next() {
                                    let ch =
                                        core::char::from_u32(u32::from(u16::from_unaligned(ule)))
                                            .unwrap();
                                    self.buffer.push(CharacterAndClass::new(ch));
                                    i += 1;
                                    if !self
                                        .decompositions
                                        .decomposition_starts_with_non_starter
                                        .contains(ch)
                                    {
                                        combining_start = i;
                                    }
                                }
                                (starter, combining_start)
                            }
                        } else {
                            let (&first, tail) = &self.decompositions.scalars32.as_ule_slice()
                                [offset..offset + len]
                                .split_first()
                                .unwrap();
                            let starter = core::char::from_u32(u32::from_unaligned(first)).unwrap();
                            if low & 0x800 == 0 {
                                // All the rest are combining
                                for &ule in tail.iter() {
                                    self.buffer.push(CharacterAndClass::new(
                                        core::char::from_u32(u32::from_unaligned(ule)).unwrap(),
                                    ));
                                }
                                (starter, 0)
                            } else {
                                let mut i = 0;
                                let mut combining_start = 0;
                                let mut it = tail.iter();
                                while let Some(&ule) = it.next() {
                                    let ch =
                                        core::char::from_u32(u32::from_unaligned(ule)).unwrap();
                                    self.buffer.push(CharacterAndClass::new(ch));
                                    i += 1;
                                    if !self
                                        .decompositions
                                        .decomposition_starts_with_non_starter
                                        .contains(ch)
                                    {
                                        combining_start = i;
                                    }
                                }
                                (starter, combining_start)
                            }
                        }
                    }
                }
            } else {
                // Hangul syllable
                // The math here comes from page 144 of Unicode 14.0
                let l = hangul_offset / HANGUL_N_COUNT;
                let v = (hangul_offset % HANGUL_N_COUNT) / HANGUL_T_COUNT;
                let t = hangul_offset % HANGUL_T_COUNT;

                self.buffer.push(CharacterAndClass::new(unsafe {
                    core::char::from_u32_unchecked(HANGUL_V_BASE + v)
                }));
                let first = unsafe { core::char::from_u32_unchecked(HANGUL_L_BASE + l) };
                if t != 0 {
                    self.buffer.push(CharacterAndClass::new(unsafe {
                        core::char::from_u32_unchecked(HANGUL_T_BASE + t)
                    }));
                    (first, 2)
                } else {
                    (first, 1)
                }
            }
        };
        debug_assert_eq!(self.pending_unnormalized_starter, None);
        while let Some(ch) = self.delegate.next() {
            if self
                .decompositions
                .decomposition_starts_with_non_starter
                .contains(ch)
            {
                if !in_inclusive_range(ch, '\u{0340}', '\u{0F81}') {
                    self.buffer.push(CharacterAndClass::new(ch));
                } else {
                    // The Tibetan special cases are starters that decompose into non-starters.
                    //
                    // Logically the canonical combining class of each special case is known
                    // at compile time, but all characters in the buffer are treated the same
                    // when looking up the canonical combining class to avoid a per-character
                    // branch that would only benefit these rare special cases.
                    match ch {
                        '\u{0340}' => {
                            // COMBINING GRAVE TONE MARK
                            self.buffer.push(CharacterAndClass::new('\u{0300}'));
                        }
                        '\u{0341}' => {
                            // COMBINING ACUTE TONE MARK
                            self.buffer.push(CharacterAndClass::new('\u{0301}'));
                        }
                        '\u{0343}' => {
                            // COMBINING GREEK KORONIS
                            self.buffer.push(CharacterAndClass::new('\u{0313}'));
                        }
                        '\u{0344}' => {
                            // COMBINING GREEK DIALYTIKA TONOS
                            self.buffer.push(CharacterAndClass::new('\u{0308}'));
                            self.buffer.push(CharacterAndClass::new('\u{0301}'));
                        }
                        '\u{0F73}' => {
                            // TIBETAN VOWEL SIGN II
                            self.buffer.push(CharacterAndClass::new('\u{0F71}'));
                            self.buffer.push(CharacterAndClass::new('\u{0F72}'));
                        }
                        '\u{0F75}' => {
                            // TIBETAN VOWEL SIGN UU
                            self.buffer.push(CharacterAndClass::new('\u{0F71}'));
                            self.buffer.push(CharacterAndClass::new('\u{0F74}'));
                        }
                        '\u{0F81}' => {
                            // TIBETAN VOWEL SIGN REVERSED II
                            self.buffer.push(CharacterAndClass::new('\u{0F71}'));
                            self.buffer.push(CharacterAndClass::new('\u{0F80}'));
                        }
                        _ => {
                            self.buffer.push(CharacterAndClass::new(ch));
                        }
                    };
                }
            } else {
                self.pending_unnormalized_starter = Some(ch);
                break;
            }
        }

        sort_slice_by_ccc(&mut self.buffer[combining_start..], self.ccc);

        Some(starter)
    }
}

pub struct DecomposingNormalizer {
    decompositions: DataPayload<CanonicalDecompositionDataV1Marker>,
    ccc: DataPayload<UnicodePropertyMapV1Marker<CanonicalCombiningClass>>,
}

impl DecomposingNormalizer {
    pub fn try_new<D>(data_provider: &D) -> Result<Self, NormalizerError>
    where
        D: ResourceProvider<CanonicalDecompositionDataV1Marker>
            + DynProvider<
                icu_properties::provider::UnicodePropertyMapV1Marker<
                    icu_properties::CanonicalCombiningClass,
                >,
            > + ?Sized,
    {
        let decompositions: DataPayload<CanonicalDecompositionDataV1Marker> = data_provider
            .load_resource(&DataRequest::default())?
            .take_payload()?;

        let ccc: DataPayload<UnicodePropertyMapV1Marker<CanonicalCombiningClass>> =
            icu_properties::maps::get_canonical_combining_class(data_provider)?;

        Ok(DecomposingNormalizer {
            decompositions: decompositions,
            ccc: ccc,
        })
    }

    pub fn normalize_iter<'data, I: Iterator<Item = char>>(
        &'data self,
        iter: I,
    ) -> Decomposition<'data, I> {
        Decomposition::new(
            iter,
            self.decompositions.get(),
            &self.ccc.get().code_point_trie,
        )
    }

    pub fn normalize(&self, text: &str) -> String {
        self.normalize_iter(text.chars()).collect()
    }

    pub fn is_normalized(&self, text: &str) -> bool {
        self.normalize_iter(text.chars()).eq(text.chars())
    }

    pub fn normalize_to<W: core::fmt::Write + ?Sized>(
        &self,
        text: &str,
        sink: &mut W,
    ) -> core::fmt::Result {
        for c in self.normalize_iter(text.chars()) {
            sink.write_char(c)?;
        }
        Ok(())
    }

    pub fn normalize_utf16(&self, text: &[u16]) -> Vec<u16> {
        let mut buf = [0u16; 2];
        let mut ret = Vec::new();
        for c in
            self.normalize_iter(decode_utf16(text.iter().copied()).map(utf16_error_to_replacement))
        {
            ret.extend_from_slice(c.encode_utf16(&mut buf));
        }
        ret
    }

    pub fn normalize_utf16_to<W: core::fmt::Write + ?Sized>(
        &self,
        text: &[u16],
        sink: &mut W,
    ) -> core::fmt::Result {
        for c in
            self.normalize_iter(decode_utf16(text.iter().copied()).map(utf16_error_to_replacement))
        {
            sink.write_char(c)?;
        }
        Ok(())
    }

    pub fn is_normalized_utf16(&self, text: &[u16]) -> bool {
        self.normalize_iter(decode_utf16(text.iter().copied()).map(utf16_error_to_replacement))
            .eq(decode_utf16(text.iter().copied()).map(utf16_error_to_replacement))
    }

    pub fn normalize_utf8(&self, text: &[u8]) -> String {
        self.normalize_iter(text.chars()).collect()
    }

    pub fn normalize_utf8_to<W: core::fmt::Write + ?Sized>(
        &self,
        text: &[u8],
        sink: &mut W,
    ) -> core::fmt::Result {
        for c in self.normalize_iter(text.chars()) {
            sink.write_char(c)?;
        }
        Ok(())
    }

    pub fn is_normalized_utf8(&self, text: &[u8]) -> bool {
        self.normalize_iter(text.chars()).eq(text.chars())
    }
}

#[cfg(test)]
mod tests {}
