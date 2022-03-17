// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use icu_codepointtrie::{CodePointTrie, CodePointTrieHeader, TrieType, TrieValue};
use icu_provider::prelude::*;
use std::convert::TryFrom;
use zerovec::ZeroVec;

#[allow(clippy::upper_case_acronyms)]
#[derive(serde::Deserialize)]
pub struct TomlCodePointTrie {
    #[serde(skip)]
    pub short_name: String,
    #[serde(skip)]
    pub long_name: String,
    #[serde(skip)]
    pub name: String,
    pub index: Vec<u16>,
    pub data_8: Option<Vec<u8>>,
    pub data_16: Option<Vec<u16>>,
    pub data_32: Option<Vec<u32>>,
    #[serde(skip)]
    pub index_length: u32,
    #[serde(skip)]
    pub data_length: u32,
    #[serde(rename = "highStart")]
    pub high_start: u32,
    #[serde(rename = "shifted12HighStart")]
    pub shifted12_high_start: u16,
    #[serde(rename = "type")]
    pub trie_type_enum_val: u8,
    #[serde(rename = "valueWidth")]
    pub value_width_enum_val: u8,
    #[serde(rename = "index3NullOffset")]
    pub index3_null_offset: u16,
    #[serde(rename = "dataNullOffset")]
    pub data_null_offset: u32,
    #[serde(rename = "nullValue")]
    pub null_value: u32,
}

impl<T: TrieValue> TryFrom<&TomlCodePointTrie> for CodePointTrie<'static, T> {
    type Error = DataError;

    fn try_from(cpt_data: &TomlCodePointTrie) -> Result<CodePointTrie<'static, T>, DataError> {
        let trie_type_enum: TrieType =
            TrieType::try_from(cpt_data.trie_type_enum_val).map_err(|e| {
                DataError::custom("Could not parse TrieType in TOML").with_display_context(&e)
            })?;
        let header = CodePointTrieHeader {
            high_start: cpt_data.high_start,
            shifted12_high_start: cpt_data.shifted12_high_start,
            index3_null_offset: cpt_data.index3_null_offset,
            data_null_offset: cpt_data.data_null_offset,
            null_value: cpt_data.null_value,
            trie_type: trie_type_enum,
        };
        let index: ZeroVec<u16> = ZeroVec::alloc_from_slice(&cpt_data.index);
        let data: Result<ZeroVec<'static, T>, T::TryFromU32Error> =
            if let Some(data_8) = &cpt_data.data_8 {
                data_8.iter().map(|i| T::try_from_u32(*i as u32)).collect()
            } else if let Some(data_16) = &cpt_data.data_16 {
                data_16.iter().map(|i| T::try_from_u32(*i as u32)).collect()
            } else if let Some(data_32) = &cpt_data.data_32 {
                data_32.iter().map(|i| T::try_from_u32(*i as u32)).collect()
            } else {
                return Err(DataError::custom(
                    "Did not find data array for CodePointTrie in TOML",
                ));
            };
        let data = data.map_err(|e| {
            DataError::custom("Could not parse data array in TOML").with_display_context(&e)
        })?;
        CodePointTrie::<T>::try_new(header, index, data).map_err(|e| {
            DataError::custom("Could not create CodePointTrie from header/index/data array in TOML")
                .with_display_context(&e)
        })
    }
}
