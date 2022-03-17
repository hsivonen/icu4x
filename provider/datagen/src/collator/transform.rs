// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

use icu_codepointtrie::CodePointTrie;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::Path;
use zerovec::ZeroVec;
use std::str::FromStr;

use crate::DatagenOptions;
use crate::uprops::codepointtrie::TomlCodePointTrie;
use crate::uprops::reader::*;
use eyre::WrapErr;
use icu_locid::LanguageIdentifier;
use icu_provider::datagen::IterableResourceProvider;
use icu_provider::prelude::*;
use writeable::Writeable;
use icu_locid::Locale;
use icu_locid::unicode_ext_key;
use icu_locid::extensions::unicode::Value;

#[derive(serde::Deserialize)]
pub struct CollationData {
    pub trie: TomlCodePointTrie,
    pub contexts: Vec<u16>,
    pub ce32s: Vec<u32>,
    // TOML integers are signed 64-bit, so the range of u64 isn't available
    pub ces: Vec<i64>,
    pub last_primaries: Vec<u16>, // length always supposed to be 4
}

#[derive(serde::Deserialize)]
pub struct CollationDiacritics {
    pub ce32s: Vec<u32>,
}

#[derive(serde::Deserialize)]
pub struct CollationJamo {
    pub ce32s: Vec<u32>,
}

#[derive(serde::Deserialize)]
pub struct CollationMetadata {
    pub bits: u32,
}

#[derive(serde::Deserialize)]
pub struct CollationReordering {
    pub min_high_no_reorder: u32,
    pub reorder_table: Vec<u8>,
    pub reorder_ranges: Vec<u32>,
}

macro_rules! collation_provider {
    ($marker:ident, $provider:ident, $key:ident, $serde_struct:ident, $suffix:literal, $conversion:expr, $toml_data:ident) => {
        use icu_collator::provider::$marker;

        pub struct $provider {
            data: HashMap<String, $serde_struct>,
        }

        impl TryFrom<&DatagenOptions> for $provider {
            type Error = DataError;
            fn try_from(options: &DatagenOptions) -> Result<Self, Self::Error> {
                Ok(Self::try_new(&options.collator_data_root.as_ref().ok_or_else(|| DataError::custom("collator data root required"))?)?)
            }
        }

        /// A data provider reading from .toml files produced by the ICU4C genrb tool.
        impl $provider {
            pub fn try_new(root_dir: &Path) -> Result<Self, DataError> {
                let data = $provider::load_data(root_dir).map_err(|e| {
                    DataError::custom("Could not create provider").with_display_context(&e)
                })?;
                Ok(Self { data })
            }

            fn load_data(root_dir: &Path) -> eyre::Result<HashMap<String, $serde_struct>> {
                let mut result = HashMap::new();
                for path in get_dir_contents(&root_dir)? {
                    let stem_bytes = if let Some(stem_bytes) = path
                        .file_stem()
                        .and_then(|p| p.to_str())
                        .ok_or_else(|| eyre::eyre!("Invalid file name: {:?}", path))?
                        .as_bytes()
                        .strip_suffix($suffix)
                    {
                        stem_bytes
                    } else {
                        continue;
                    };
                    let mut key = String::from_utf8(stem_bytes.to_vec())
                        .wrap_err_with(|| "Non-UTF-8 file name")?;
                    let toml_str = read_path_to_string(&path)?;
                    let toml_obj: $serde_struct = toml::from_str(&toml_str)
                        .wrap_err_with(|| format!("Could not parse TOML: {:?}", path))?;
                    key.make_ascii_lowercase();
                    result.insert(key, toml_obj);
                }
                Ok(result)
            }
        }

        impl ResourceProvider<$marker> for $provider {
            fn load_resource(&self, req: &DataRequest) -> Result<DataResponse<$marker>, DataError> {
                let langid = req.options.get_langid();
                let mut s = String::new();
                langid.write_to(&mut s).map_err(|_| {
                    DataErrorKind::MissingResourceOptions
                        .with_req(icu_collator::provider::key::$key, req)
                })?;
                if &s == "und" {
                    s = String::from("root");
                } else {
                    // No safe method for in-place replacement.
                    s = s.replace('-', "_");
                    s.make_ascii_lowercase();
                }
                if let Some(extension) = &req.options.get_unicode_ext(&unicode_ext_key!("co")) {
                    let extension_string = extension.to_string();
                    let extension_str = &extension_string[..];
                    s.push('_');
                    match extension_str {
                        "trad" => {
                            s.push_str("traditional");
                        }
                        "phonebk" => {
                            s.push_str("phonebook");
                        }
                        "dict" => {
                            s.push_str("dictionary");
                        }
                        "gb2312" => {
                            s.push_str("gb2312han");
                        }
                        _ => {
                            s.push_str(extension_str);
                        }
                    }
                } else {
                    // "standard" is the default for all but two languages: sv and zh.
                    // Since there are only two special cases, hard-coding them
                    // here for now instead of making the defaulting fancy and data driven.
                    // The Swedish naming seems ad hoc from
                    // https://unicode-org.atlassian.net/browse/CLDR-679 .

                    if langid.language == "zh" {
                        s.push_str("_pinyin");
                    } else if langid.language == "sv" {
                        s.push_str("_reformed");
                    } else {
                        s.push_str("_standard");
                    }
                }

                if let Some($toml_data) = self.data.get(&s) {
                    Ok(DataResponse {
                        metadata: DataResponseMetadata::default(),
                        // The struct conversion is macro-based instead of
                        // using a method on the Serde struct, because the
                        // method approach caused lifetime issues that I
                        // didn't know how to solve.
                        payload: Some(DataPayload::from_owned($conversion)),
                    })
                } else {
                    Err(DataErrorKind::MissingResourceOptions
                        .with_req(icu_collator::provider::key::$key, req))
                }
            }
        }

        icu_provider::impl_dyn_provider!(
            $provider,
            [$marker,],
            SERDE_SE,
            ITERABLE_SERDE_SE,
            DATA_CONVERTER
        );

        impl IterableResourceProvider<$marker> for $provider {
            fn supported_options(
                &self,
            ) -> Result<Box<dyn Iterator<Item = ResourceOptions>>, DataError> {
                let list: Vec<ResourceOptions> = self
                    .data
                    .keys()
                    .map(|k| {
                        let (language, variant) = k.rsplit_once('_').unwrap();
                        let langid = if language == "root" {
                            LanguageIdentifier::default()
                        } else {
                            language.parse().unwrap()
                        };
                        let mut locale = Locale::from(langid);
                        // See above for the two special cases.
                        if !((language == "zh" && variant == "pinyin")
                            || (language == "sv" && variant == "reformed")
                            || ((language != "zh" && language != "sv") && variant == "standard"))
                        {
                            let shortened = match variant {
                                "traditional" => "trad",
                                "phonebook" => "phonebk",
                                "dictionary" => "dict",
                                "gb2312han" => "gb2312",
                                _ => variant,
                            };
                            // TODO: Is there a compile-time macro for this?
                            let key = "co".parse::<icu_locid::extensions::unicode::Key>().unwrap();
                            locale.extensions.unicode.keywords.set(key, Value::from_str(shortened).expect("valid extension subtag"));
                        };
                        ResourceOptions::from(locale)
                    })
                    .collect();
                Ok(Box::new(list.into_iter()))
            }
        }
    };
}

collation_provider!(
    CollationDataV1Marker,
    CollationDataDataProvider,
    COLLATION_DATA_KEY,
    CollationData,
    b"_data",
    icu_collator::provider::CollationDataV1 {
        trie: CodePointTrie::<u32>::try_from(&toml_data.trie)?,
        contexts: ZeroVec::alloc_from_slice(&toml_data.contexts),
        ce32s: ZeroVec::alloc_from_slice(&toml_data.ce32s),
        ces: toml_data.ces.iter().map(|i| *i as u64).collect(),
        last_primaries: ZeroVec::alloc_from_slice(&toml_data.last_primaries),
    },
    toml_data
);

collation_provider!(
    CollationDiacriticsV1Marker,
    CollationDiacriticsDataProvider,
    COLLATION_DIACRITICS_KEY,
    CollationDiacritics,
    b"_dia",
    icu_collator::provider::CollationDiacriticsV1 {
        ce32s: ZeroVec::alloc_from_slice(&toml_data.ce32s),
    },
    toml_data
);

collation_provider!(
    CollationJamoV1Marker,
    CollationJamoDataProvider,
    COLLATION_JAMO_KEY,
    CollationJamo,
    b"_jamo",
    icu_collator::provider::CollationJamoV1 {
        ce32s: ZeroVec::alloc_from_slice(&toml_data.ce32s),
    },
    toml_data
);

collation_provider!(
    CollationMetadataV1Marker,
    CollationMetadataDataProvider,
    COLLATION_METADATA_KEY,
    CollationMetadata,
    b"_meta",
    icu_collator::provider::CollationMetadataV1 {
        bits: toml_data.bits,
    },
    toml_data
);

collation_provider!(
    CollationReorderingV1Marker,
    CollationReorderingDataProvider,
    COLLATION_REORDERING_KEY,
    CollationReordering,
    b"_reord",
    icu_collator::provider::CollationReorderingV1 {
        min_high_no_reorder: toml_data.min_high_no_reorder,
        reorder_table: ZeroVec::alloc_from_slice(&toml_data.reorder_table),
        reorder_ranges: ZeroVec::alloc_from_slice(&toml_data.reorder_ranges),
    },
    toml_data
);
