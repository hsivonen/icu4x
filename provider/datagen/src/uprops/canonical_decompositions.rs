// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use crate::DatagenOptions;
use crate::uprops::decompositions_serde;
use crate::uprops::reader::*;
use icu_codepointtrie::CodePointTrie;
use std::convert::TryFrom;
use zerovec::ZeroVec;

use icu_normalizer::provider::CanonicalDecompositionDataV1;
use icu_normalizer::provider::CanonicalDecompositionDataV1Marker;
use icu_provider::datagen::IterableResourceProvider;
use icu_provider::prelude::*;
use icu_uniset::UnicodeSetBuilder;

use std::path::Path;

pub struct CanonicalDecompositionDataProvider {
    data: decompositions_serde::CanonicalDecompositionData,
}

impl TryFrom<&DatagenOptions> for CanonicalDecompositionDataProvider {
    type Error = DataError;
    fn try_from(options: &DatagenOptions) -> Result<Self, Self::Error> {
        Ok(Self::try_new(&options.uprops_root.as_ref().ok_or_else(|| DataError::custom("uprops data root required"))?)?)
    }
}

/// A data provider reading from .toml files produced by the ICU4C icudataexport tool.
impl CanonicalDecompositionDataProvider {
    pub fn try_new(root_dir: &Path) -> Result<Self, DataError> {
        let path = root_dir.join("decompositions.toml");
        let toml_str = read_path_to_string(&path).map_err(|e| {
            DataError::custom("Could not create provider").with_display_context(&e)
        })?;
        let toml_obj: decompositions_serde::CanonicalDecompositionData = toml::from_str(&toml_str).map_err(|e| {
            DataError::custom("Could not parse TOML").with_display_context(&e)
        })?;

        Ok(Self { data: toml_obj })
    }
}

impl ResourceProvider<CanonicalDecompositionDataV1Marker> for CanonicalDecompositionDataProvider {
    fn load_resource(
        &self,
        _req: &DataRequest,
    ) -> Result<DataResponse<CanonicalDecompositionDataV1Marker>, DataError> {
        let toml_data = &self.data;

        let mut builder = UnicodeSetBuilder::new();
        for range in &toml_data.ranges {
            builder.add_range_u32(&(range.0..=range.1));
        }
        let uniset = builder.build();

        let trie = CodePointTrie::<u32>::try_from(&toml_data.trie);

        Ok(DataResponse {
            metadata: DataResponseMetadata::default(),
            payload: Some(DataPayload::from_owned(CanonicalDecompositionDataV1 {
                trie: trie?,
                scalars16: ZeroVec::alloc_from_slice(&toml_data.scalars16),
                scalars32: ZeroVec::alloc_from_slice(&toml_data.scalars32),
                decomposition_starts_with_non_starter: uniset,
            })),
        })
    }
}

icu_provider::impl_dyn_provider!(
    CanonicalDecompositionDataProvider,
    [CanonicalDecompositionDataV1Marker,],
    SERDE_SE,
    ITERABLE_SERDE_SE,
    DATA_CONVERTER
);

impl IterableResourceProvider<CanonicalDecompositionDataV1Marker>
    for CanonicalDecompositionDataProvider
{
    fn supported_options(&self) -> Result<Box<dyn Iterator<Item = ResourceOptions>>, DataError> {
        Ok(Box::new(core::iter::once(ResourceOptions::default())))
    }
}
