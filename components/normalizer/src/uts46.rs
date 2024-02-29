// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

//! Bundles the part of UTS 46 that makes sense to implement as a
//! normalization with some tightly-coupled data that isn't technically
//! quite part of normalization data as matter of implementation details.
//!
//! This is meant to be used as a building block of an UTS 46
//! implementation, such as `icu_idna`.

use crate::provider::Uts46SetsV1Marker;
use crate::CanonicalCompositionsV1Marker;
use crate::CanonicalDecompositionDataV1Marker;
use crate::CanonicalDecompositionTablesV1Marker;
use crate::CompatibilityDecompositionTablesV1Marker;
use crate::ComposingNormalizer;
use crate::NormalizerError;
use crate::Uts46DecompositionSupplementV1Marker;
use icu_provider::DataPayload;
use icu_provider::DataProvider;

/// A mapper that knows how to performs the subset of UTS 46 processing
/// documented on the [`Uts46Mapper::map_iter`] method.
#[derive(Debug)]
pub struct Uts46Mapper {
    normalizer: ComposingNormalizer,
    sets: DataPayload<Uts46SetsV1Marker>,
}

impl Uts46Mapper {
    /// See [`Self::try_new_uts46_without_ignored_and_disallowed_unstable`].
    #[cfg(all(feature = "experimental", feature = "compiled_data"))]
    pub const fn new() -> Self {
        Uts46Mapper {
            normalizer: ComposingNormalizer::new_uts46_without_ignored_and_disallowed(),
            sets: DataPayload::from_static_ref(
                crate::provider::Baked::SINGLETON_NORMALIZER_UTS46SETS_V1,
            ),
        }
    }

    /// 🚧 \[Experimental\] UTS 46 constructor
    ///
    /// NOTE: This method remains experimental until suitability of this feature as part of
    /// IDNA processing has been demonstrated.
    ///
    /// <div class="stab unstable">
    /// 🚧 This code is experimental; it may change at any time, in breaking or non-breaking ways,
    /// including in SemVer minor releases. It can be enabled with the "experimental" Cargo feature
    /// of the icu meta-crate. Use with caution.
    /// <a href="https://github.com/unicode-org/icu4x/issues/2614">#2614</a>
    /// </div>
    #[cfg(feature = "experimental")]
    pub fn try_new<D>(provider: &D) -> Result<Self, NormalizerError>
    where
        D: DataProvider<CanonicalDecompositionDataV1Marker>
            + DataProvider<Uts46DecompositionSupplementV1Marker>
            + DataProvider<CanonicalDecompositionTablesV1Marker>
            + DataProvider<CompatibilityDecompositionTablesV1Marker>
            // UTS 46 tables merged into CompatibilityDecompositionTablesV1Marker
            + DataProvider<CanonicalCompositionsV1Marker>
            + DataProvider<Uts46SetsV1Marker>
            + ?Sized,
    {
        let normalizer =
            ComposingNormalizer::try_new_uts46_without_ignored_and_disallowed_unstable(provider)?;

        let sets: DataPayload<Uts46SetsV1Marker> =
            provider.load(Default::default())?.take_payload()?;

        Ok(Uts46Mapper { normalizer, sets })
    }
}

impl Uts46Mapper
{
    /// Returns an iterator adaptor that turns an `Iterator` over `char`
    /// into an iterator yielding a `char` sequence that gets the following
    /// operations from the "Processing" section of UTS 46 lazily applied
    /// to it:
    ///
    /// 1. The _ignored_ characters are ignored if `ignored_as_errors` is
    ///    `false` and turned into U+FFFD if `ignored_as_errors` is `true`.
    ///    (Pass `false` for the "Map" phase and `true` for the "Validate"
    ///    phase.)
    /// 2. The _mapped_ characters are mapped.
    /// 3. The _disallowed_ characters are replaced with U+FFFD,
    ///    which itself is a disallowed character.
    /// 4. The result is normalized to NFC.
    ///
    /// Notably:
    ///
    /// * ASCII characters disallowed by STD3 or the WHATWG URL Standard
    ///   are _not_ turned into U+FFFD. STD3 or WHATWG ASCII deny list
    ///   should be implemented as a post-processing step.
    /// * Transitional processing is not performed. Transitional mapping
    ///   would be a pre-processing step, but transitional processing is
    ///   deprecated, and none of Firefox, Safari, or Chrome use it.
    pub fn map_iter<'delegate, I: Iterator<Item = char> + 'delegate>(
        &'delegate self,
        iter: I,
        ignored_as_errors: bool,
    ) -> impl Iterator<Item = char> + 'delegate {
        let sets = self.sets.get();
        self.normalizer.normalize_iter(iter.filter_map(move |c| {
            if sets.ignored.contains(c) {
                if ignored_as_errors {
                    Some('\u{FFFD}')
                } else {
                    None
                }
            } else if sets.disallowed.contains(c) {
                Some('\u{FFFD}')
            } else {
                Some(c)
            }
        }))
    }
}
