// @generated
/// Implement `DataProvider<GregorianYearSymbolsV1Marker>` on the given struct using the data
/// hardcoded in this file. This allows the struct to be used with
/// `icu`'s `_unstable` constructors.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_datetime_symbols_gregory_years_v1 {
    ($ provider : ty) => {
        #[clippy::msrv = "1.67"]
        const _: () = <$provider>::MUST_USE_MAKE_PROVIDER_MACRO;
        #[clippy::msrv = "1.67"]
        impl icu_provider::DataProvider<icu::datetime::provider::neo::GregorianYearSymbolsV1Marker> for $provider {
            fn load(&self, req: icu_provider::DataRequest) -> Result<icu_provider::DataResponse<icu::datetime::provider::neo::GregorianYearSymbolsV1Marker>, icu_provider::DataError> {
                static MR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0!\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\xA8\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA4\xA8") })
                });
                static PS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0#\0\xD9\x84\xD9\x87 \xD9\x85\xDB\x8C\xD9\x84\xD8\xA7\xD8\xAF \xDA\x85\xD8\xAE\xD9\x87 \xD9\x88\xDA\x93\xD8\xA7\xD9\x86\xD8\xAF\xDB\x90\xD9\x84\xD9\x87 \xD9\x85\xDB\x8C\xD9\x84\xD8\xA7\xD8\xAF \xDA\x85\xD8\xAE\xD9\x87 \xD9\x88\xD8\xB1\xD9\x88\xD8\xB3\xD8\xAA\xD9\x87") })
                });
                static KOK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF.\xE0\xA4\xB6.") })
                });
                static KOK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x95\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBF\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xB6\xE0\xA4\x95") })
                });
                static BN_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\xB8\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xAA\xE0\xA7\x82\xE0\xA6\xB0\xE0\xA7\x8D\xE0\xA6\xAC\xE0\xA6\x96\xE0\xA7\x83\xE0\xA6\xB7\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xBE\xE0\xA6\xAC\xE0\xA7\x8D\xE0\xA6\xA6") })
                });
                static BN_IN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\xB8\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xAA\xE0\xA7\x82\xE0\xA6\xB0\xE0\xA7\x8D\xE0\xA6\xAC\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\xB7\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xBE\xE0\xA6\xAC\xE0\xA7\x8D\xE0\xA6\xA6") })
                });
                static BN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\xB8\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xAA\xE0\xA7\x82\xE0\xA6\xB0\xE0\xA7\x8D\xE0\xA6\xAC\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA7\x80\xE0\xA6\xB7\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xBE\xE0\xA6\xAC\xE0\xA7\x8D\xE0\xA6\xA6") })
                });
                static AS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA7\xB0\xE0\xA7\x80\xE0\xA6\xB7\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xAA\xE0\xA7\x82\xE0\xA7\xB0\xE0\xA7\x8D\xE0\xA6\xAC\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA7\xB0\xE0\xA7\x80\xE0\xA6\xB7\xE0\xA7\x8D\xE0\xA6\x9F\xE0\xA6\xBE\xE0\xA6\xAC\xE0\xA7\x8D\xE0\xA6\xA6") })
                });
                static OR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0$\0\xE0\xAC\x96\xE0\xAD\x8D\xE0\xAC\xB0\xE0\xAD\x80\xE0\xAC\xB7\xE0\xAD\x8D\xE0\xAC\x9F\xE0\xAC\xAA\xE0\xAD\x82\xE0\xAC\xB0\xE0\xAD\x8D\xE0\xAC\xAC\xE0\xAC\x96\xE0\xAD\x8D\xE0\xAC\xB0\xE0\xAD\x80\xE0\xAC\xB7\xE0\xAD\x8D\xE0\xAC\x9F\xE0\xAC\xBE\xE0\xAC\xAC\xE0\xAD\x8D\xE0\xAC\xA6") })
                });
                static GU_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0%\0\xE0\xAA\x88\xE0\xAA\xB8\xE0\xAA\xB5\xE0\xAB\x80\xE0\xAA\xB8\xE0\xAA\xA8 \xE0\xAA\xAA\xE0\xAB\x82\xE0\xAA\xB0\xE0\xAB\x8D\xE0\xAA\xB5\xE0\xAB\x87\xE0\xAA\x87\xE0\xAA\xB8\xE0\xAA\xB5\xE0\xAB\x80\xE0\xAA\xB8\xE0\xAA\xA8") })
                });
                static KN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0%\0\xE0\xB2\x95\xE0\xB3\x8D\xE0\xB2\xB0\xE0\xB2\xBF\xE0\xB2\xB8\xE0\xB3\x8D\xE0\xB2\xA4 \xE0\xB2\xAA\xE0\xB3\x82\xE0\xB2\xB0\xE0\xB3\x8D\xE0\xB2\xB5\xE0\xB2\x95\xE0\xB3\x8D\xE0\xB2\xB0\xE0\xB2\xBF\xE0\xB2\xB8\xE0\xB3\x8D\xE0\xB2\xA4 \xE0\xB2\xB6\xE0\xB2\x95") })
                });
                static KY_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0&\0\xD0\xB1\xD0\xB8\xD0\xB7\xD0\xB4\xD0\xB8\xD0\xBD \xD0\xB7\xD0\xB0\xD0\xBC\xD0\xB0\xD0\xBD\xD0\xB3\xD0\xB0 \xD1\x87\xD0\xB5\xD0\xB9\xD0\xB8\xD0\xBD\xD0\xB1\xD0\xB8\xD0\xB7\xD0\xB4\xD0\xB8\xD0\xBD \xD0\xB7\xD0\xB0\xD0\xBC\xD0\xB0\xD0\xBD") })
                });
                static RU_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0(\0\xD0\xB4\xD0\xBE \xD0\xA0\xD0\xBE\xD0\xB6\xD0\xB4\xD0\xB5\xD1\x81\xD1\x82\xD0\xB2\xD0\xB0 \xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0\xD0\xBE\xD1\x82 \xD0\xA0\xD0\xBE\xD0\xB6\xD0\xB4\xD0\xB5\xD1\x81\xD1\x82\xD0\xB2\xD0\xB0 \xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0") })
                });
                static CHR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0)\0\xE1\x8F\xA7\xE1\x8F\x93\xE1\x8E\xB7\xE1\x8E\xB8 \xE1\x8E\xA4\xE1\x8E\xB7\xE1\x8E\xAF\xE1\x8F\x8D\xE1\x8F\x97 \xE1\x8E\xA6\xE1\x8E\xB6\xE1\x8F\x81\xE1\x8F\x9B\xE1\x8E\xA0\xE1\x8F\x83 \xE1\x8F\x99\xE1\x8E\xBB\xE1\x8F\x82") })
                });
                static BE_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0*\0\xD0\xB4\xD0\xB0 \xD0\xBD\xD0\xB0\xD1\x80\xD0\xB0\xD0\xB4\xD0\xB6\xD1\x8D\xD0\xBD\xD0\xBD\xD1\x8F \xD0\xA5\xD1\x80\xD1\x8B\xD1\x81\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0\xD0\xB0\xD0\xB4 \xD0\xBD\xD0\xB0\xD1\x80\xD0\xB0\xD0\xB4\xD0\xB6\xD1\x8D\xD0\xBD\xD0\xBD\xD1\x8F \xD0\xA5\xD1\x80\xD1\x8B\xD1\x81\xD1\x82\xD0\xBE\xD0\xB2\xD0\xB0") })
                });
                static TH_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0*\0\xE0\xB8\x9B\xE0\xB8\xB5\xE0\xB8\x81\xE0\xB9\x88\xE0\xB8\xAD\xE0\xB8\x99\xE0\xB8\x84\xE0\xB8\xA3\xE0\xB8\xB4\xE0\xB8\xAA\xE0\xB8\x95\xE0\xB8\x81\xE0\xB8\xB2\xE0\xB8\xA5\xE0\xB8\x84\xE0\xB8\xA3\xE0\xB8\xB4\xE0\xB8\xAA\xE0\xB8\x95\xE0\xB9\x8C\xE0\xB8\xA8\xE0\xB8\xB1\xE0\xB8\x81\xE0\xB8\xA3\xE0\xB8\xB2\xE0\xB8\x8A") })
                });
                static TE_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0+\0\xE0\xB0\x95\xE0\xB1\x8D\xE0\xB0\xB0\xE0\xB1\x80\xE0\xB0\xB8\xE0\xB1\x8D\xE0\xB0\xA4\xE0\xB1\x81 \xE0\xB0\xAA\xE0\xB1\x82\xE0\xB0\xB0\xE0\xB1\x8D\xE0\xB0\xB5\xE0\xB0\x82\xE0\xB0\x95\xE0\xB1\x8D\xE0\xB0\xB0\xE0\xB1\x80\xE0\xB0\xB8\xE0\xB1\x8D\xE0\xB0\xA4\xE0\xB1\x81 \xE0\xB0\xB6\xE0\xB0\x95\xE0\xB0\x82") })
                });
                static SI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0+\0\xE0\xB6\x9A\xE0\xB7\x8A\xE2\x80\x8D\xE0\xB6\xBB\xE0\xB7\x92\xE0\xB7\x83\xE0\xB7\x8A\xE0\xB6\xAD\xE0\xB7\x94 \xE0\xB6\xB4\xE0\xB7\x96\xE0\xB6\xBB\xE0\xB7\x8A\xE0\xB7\x80\xE0\xB6\x9A\xE0\xB7\x8A\xE2\x80\x8D\xE0\xB6\xBB\xE0\xB7\x92\xE0\xB7\x83\xE0\xB7\x8A\xE0\xB6\xAD\xE0\xB7\x94 \xE0\xB7\x80\xE0\xB6\xBB\xE0\xB7\x8A\xE0\xB7\x82") })
                });
                static CV_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0,\0\xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xBE\xD1\x81 \xD2\xAB\xD1\x83\xD1\x80\xD0\xB0\xD0\xBB\xD0\xBD\xD3\x91 \xD0\xBA\xD1\x83\xD0\xBD\xD1\x87\xD1\x87\xD0\xB5\xD0\xBD\xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xBE\xD1\x81 \xD2\xAB\xD1\x83\xD1\x80\xD0\xB0\xD0\xBB\xD0\xBD\xD3\x91 \xD0\xBA\xD1\x83\xD0\xBD\xD1\x80\xD0\xB0\xD0\xBD") })
                });
                static KK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0.\0\xD0\x91\xD1\x96\xD0\xB7\xD0\xB4\xD1\x96\xD2\xA3 \xD0\xB7\xD0\xB0\xD0\xBC\xD0\xB0\xD0\xBD\xD1\x8B\xD0\xBC\xD1\x8B\xD0\xB7\xD2\x93\xD0\xB0 \xD0\xB4\xD0\xB5\xD0\xB9\xD1\x96\xD0\xBD\xD0\xB1\xD1\x96\xD0\xB7\xD0\xB4\xD1\x96\xD2\xA3 \xD0\xB7\xD0\xB0\xD0\xBC\xD0\xB0\xD0\xBD\xD1\x8B\xD0\xBC\xD1\x8B\xD0\xB7") })
                });
                static BRX_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0.\0\xE0\xA4\x96\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA4\xBE\xE0\xA4\x87\xE0\xA4\xB7\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA4\xA8\xE0\xA4\xBF \xE0\xA4\xB8\xE0\xA4\xBF\xE0\xA4\x97\xE0\xA4\xBE\xE0\xA4\x82\xE0\xA4\x86\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA8\xE2\x80\x99 \xE0\xA4\xA6\xE0\xA4\xBE\xE0\xA4\xAE\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA4\xBF") })
                });
                static ML_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0:\0\xE0\xB4\x95\xE0\xB5\x8D\xE0\xB4\xB0\xE0\xB4\xBF\xE0\xB4\xB8\xE0\xB5\x8D\xE2\x80\x8C\xE0\xB4\xA4\xE0\xB5\x81\xE0\xB4\xB5\xE0\xB4\xBF\xE0\xB4\xA8\xE0\xB5\x8D \xE0\xB4\xAE\xE0\xB5\x81\xE0\xB4\xAE\xE0\xB5\x8D\xE0\xB4\xAA\xE0\xB5\x8D\xE0\xB4\x86\xE0\xB4\xA8\xE0\xB5\x8D\xE0\xB4\xA8\xE0\xB5\x8B \xE0\xB4\xA1\xE0\xB5\x8A\xE0\xB4\xAE\xE0\xB4\xBF\xE0\xB4\xA8\xE0\xB4\xBF") })
                });
                static MY_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0:\0\xE1\x80\x81\xE1\x80\x9B\xE1\x80\x85\xE1\x80\xBA\xE1\x80\x90\xE1\x80\xB1\xE1\x80\xAC\xE1\x80\xBA \xE1\x80\x99\xE1\x80\x95\xE1\x80\xB1\xE1\x80\xAB\xE1\x80\xBA\xE1\x80\x99\xE1\x80\xAE\xE1\x80\x94\xE1\x80\xBE\xE1\x80\x85\xE1\x80\xBA\xE1\x80\x81\xE1\x80\x9B\xE1\x80\x85\xE1\x80\xBA\xE1\x80\x94\xE1\x80\xBE\xE1\x80\x85\xE1\x80\xBA") })
                });
                static FF_ADLM_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0:\0\xF0\x9E\xA4\x80\xF0\x9E\xA4\xA3\xF0\x9E\xA4\xAE \xF0\x9E\xA4\x80\xF0\x9E\xA4\xB2\xF0\x9E\xA5\x86\xF0\x9E\xA4\xA2\xF0\x9E\xA4\xA6\xF0\x9E\xA4\xAD \xF0\x9E\xA4\x8B\xF0\x9E\xA5\x85\xF0\x9E\xA4\xA7\xF0\x9E\xA4\xA2\xF0\x9E\xA5\x84\xF0\x9E\xA4\x87\xF0\x9E\xA4\xA2\xF0\x9E\xA5\x84\xF0\x9E\xA4\xB1\xF0\x9E\xA4\xAE \xF0\x9E\xA4\x80\xF0\x9E\xA4\xB2\xF0\x9E\xA5\x86\xF0\x9E\xA4\xA2\xF0\x9E\xA4\xA6\xF0\x9E\xA4\xAD \xF0\x9E\xA4\x8B\xF0\x9E\xA5\x85\xF0\x9E\xA4\xA7\xF0\x9E\xA4\xA2\xF0\x9E\xA5\x84") })
                });
                static TT_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\"\0\xD0\xB1\xD0\xB5\xD0\xB7\xD0\xBD\xD0\xB5\xD2\xA3 \xD1\x8D\xD1\x80\xD0\xB0\xD0\xB3\xD0\xB0 \xD0\xBA\xD0\xB0\xD0\xB4\xD3\x99\xD1\x80\xD0\xBC\xD0\xB8\xD0\xBB\xD0\xB0\xD0\xB4\xD0\xB8") })
                });
                static MN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\"\0\xD0\xBC\xD0\xB0\xD0\xBD\xD0\xB0\xD0\xB9 \xD1\x8D\xD1\x80\xD0\xB8\xD0\xBD\xD0\xB8\xD0\xB9 \xD3\xA9\xD0\xBC\xD0\xBD\xD3\xA9\xD1\x85\xD0\xBC\xD0\xB0\xD0\xBD\xD0\xB0\xD0\xB9 \xD1\x8D\xD1\x80\xD0\xB8\xD0\xBD\xD0\xB8\xD0\xB9") })
                });
                static BG_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\n\0\xD0\xBF\xD1\x80.\xD0\xA5\xD1\x80.\xD1\x81\xD0\xBB.\xD0\xA5\xD1\x80.") })
                });
                static FO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\n\0fyri Kristeftir Krist") })
                });
                static CS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\n\0p\xC5\x99. n. l.n. l.") })
                });
                static HSB_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\n\0p\xC5\x99.Chr.n.po Chr.n.") })
                });
                static DSB_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\n\0p\xC5\x9B.Chr.n.p\xC3\xB3 Chr.n.") })
                });
                static EN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0Before ChristAnno Domini") })
                });
                static QU_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0\xC3\xB1awpa cristuchanta cristu") })
                });
                static RM_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0avant Cristussuenter Cristus") })
                });
                static IT_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0avanti Cristodopo Cristo") })
                });
                static EN_CA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0before ChristAnno Domini") })
                });
                static ET_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0enne Kristustp\xC3\xA4rast Kristust") })
                });
                static SV_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0f\xC3\xB6re Kristusefter Kristus") })
                });
                static SQ_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0para Krishtitmbas Krishtit") })
                });
                static AF_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\r\0voor Christusna Christus") })
                });
                static KGP_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0Cristo joCristo kar k\xE1\xBB\xB9") })
                });
                static CY_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0Cyn CristOed Crist") })
                });
                static KK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD0\xB1.\xD0\xB7.\xD0\xB4.\xD0\xB1.\xD0\xB7.") })
                });
                static KY_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD0\xB1.\xD0\xB7.\xD1\x87.\xD0\xB1.\xD0\xB7.") })
                });
                static TT_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD0\xB1.\xD1\x8D.\xD0\xBA.\xD0\xBC\xD0\xB8\xD0\xBB\xD0\xB0\xD0\xB4\xD0\xB8") })
                });
                static BS_CYRL_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD0\xBF.\xD0\xBD.\xD0\xB5.\xD0\xBD.\xD0\xB5.") })
                });
                static HY_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD5\xB4.\xD5\xA9.\xD5\xA1.\xD5\xB4.\xD5\xA9.") })
                });
                static KS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xD8\xA8\xDB\x8C \xD8\xB3\xDB\x8C\xD8\xA7\xDB\x92 \xDA\x88\xDB\x8C") })
                });
                static ZH_HK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xE5\x85\xAC\xE5\x85\x83\xE5\x89\x8D\xE5\x85\xAC\xE5\x85\x83") })
                });
                static JA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xE7\xB4\x80\xE5\x85\x83\xE5\x89\x8D\xE8\xA5\xBF\xE6\x9A\xA6") })
                });
                static YUE_HANS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xE8\xA5\xBF\xE5\x85\x83\xE5\x89\x8D\xE8\xA5\xBF\xE5\x85\x83") })
                });
                static KO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0\xEA\xB8\xB0\xEC\x9B\x90\xEC\xA0\x84\xEC\x84\x9C\xEA\xB8\xB0") })
                });
                static FR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\t\0av. J.-C.ap. J.-C.") })
                });
                static EN_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x01\0BA") })
                });
                static CY_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x01\0CO") })
                });
                static GD_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x01\0RA") })
                });
                static EU_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x01\0ao") })
                });
                static KEA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0AKDK") })
                });
                static CEB_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0BCAD") })
                });
                static SD_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0BCCD") })
                });
                static PCM_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0BKKIY") })
                });
                static CY_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0CCOC") })
                });
                static WO_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0JCAD") })
                });
                static SW_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0KKBK") })
                });
                static TO_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0KMTS") })
                });
                static GA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0RCAD") })
                });
                static ID_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0SMM") })
                });
                static FA_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0\xD9\x82\xD9\x85") })
                });
                static CA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0aCdC") })
                });
                static YO_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0BCEAD") })
                });
                static UND_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0BCECE") })
                });
                static HA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0K.HBHAI") })
                });
                static TR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0M\xC3\x96MS") })
                });
                static VI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0TCNCN") })
                });
                static FI_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0eKrjKr") })
                });
                static ET_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0eKrpKr") })
                });
                static DA_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0fKreKr") })
                });
                static HU_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0ie.isz.") })
                });
                static KGP_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0C.j.C.kk.") })
                });
                static EU_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0K.a.K.o.") })
                });
                static EU_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0K.a.Kristo ondoren") })
                });
                static YRL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0K.s.K.a.") })
                });
                static MS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0S.M.TM") })
                });
                static IG_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0T.K.A.K.") })
                });
                static ES_419_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0a.C.d.C.") })
                });
                static SC_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0a.C.p.C.") })
                });
                static QU_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0a.d.d.C.") })
                });
                static QU_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0a.d.dC") })
                });
                static AST_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0e.C.d.C.") })
                });
                static FI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0eKr.jKr.") })
                });
                static IS_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0f.k.e.k.") })
                });
                static UZ_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0m.a.milodiy") })
                });
                static SQ_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0p.K.mb.K.") })
                });
                static AF_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x04\0v.C.n.C.") })
                });
                static AR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x05\0\xD9\x82.\xD9\x85\xD9\x85") })
                });
                static ES_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x05\0a. C.d. C.") })
                });
                static AZ_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x05\0e.\xC9\x99.y.e.") })
                });
                static DA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x05\0f.Kr.e.Kr.") })
                });
                static HU_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x05\0i. e.i. sz.") })
                });
                static RO_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xC3\xAE.Hr.d.Hr.") })
                });
                static EL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xCF\x80.\xCE\xA7.\xCE\xBC.\xCE\xA7.") })
                });
                static MN_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xD0\x9C\xD0\xAD\xD3\xA8\xD0\x9C\xD0\xAD") })
                });
                static TG_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xD0\x9F\xD0\xB5\xD0\x9C\xD0\x9F\xD0\xB0\xD0\x9C") })
                });
                static UZ_CYRL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xD0\xBC.\xD0\xB0.\xD0\xBC\xD0\xB8\xD0\xBB\xD0\xBE\xD0\xB4\xD0\xB8\xD0\xB9") })
                });
                static FA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0\xD9\x82.\xD9\x85.\xD9\x85.") })
                });
                static IA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0a.Chr.p.Chr.") })
                });
                static WO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0av. JCAD") })
                });
                static BS_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0p.n.e.n. e.") })
                });
                static PL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0p.n.e.n.e.") })
                });
                static VI_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0tr. CNsau CN") })
                });
                static NL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0v.Chr.n.Chr.") })
                });
                static CV_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0\xD0\xBF. \xD1\x8D.\xD1\x85. \xD1\x8D.") })
                });
                static AM_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0\xE1\x8B\x93/\xE1\x8B\x93\xE1\x8B\x93/\xE1\x88\x9D") })
                });
                static RM_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0av. Cr.s. Cr.") })
                });
                static LV_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0p.m.\xC4\x93.m.\xC4\x93.") })
                });
                static LT_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0pr. Kr.po Kr.") })
                });
                static HR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0pr. Kr.po. Kr.") })
                });
                static HR_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0pr.n.e.AD") })
                });
                static DE_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x07\0v. Chr.n. Chr.") })
                });
                static TK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0B.e.\xC3\xB6\xC5\x88B.e.") })
                });
                static HE_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0\xD7\x9C\xD7\xA4\xD7\xA0\xD7\x99\xD7\x90\xD7\x97\xD7\xA8\xD7\x99\xD7\x99") })
                });
                static TO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0ki mu\xCA\xBBata\xCA\xBBu \xCA\xBBo S\xC4\xABs\xC5\xAB") })
                });
                static BS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0p. n. e.n. e.") })
                });
                static CS_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0p\xC5\x99.n.l.n.l.") })
                });
                static SK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x08\0pred Kr.po Kr.") })
                });
                static TK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0Isadan \xC3\xB6\xC5\x88Isadan so\xC5\x88") })
                });
                static YO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0Saju KristiLehin Kristi") })
                });
                static IG_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0Tupu KraistAf\xE1\xBB\x8D Kra\xE1\xBB\x8Bst") })
                });
                static BE_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xD0\xB4\xD0\xB0 \xD0\xBD.\xD1\x8D.\xD0\xBD.\xD1\x8D.") })
                });
                static UK_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xD0\xB4\xD0\xBE \xD0\xBD.\xD0\xB5.\xD0\xBD.\xD0\xB5.") })
                });
                static RU_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xD0\xB4\xD0\xBE \xD0\xBD.\xD1\x8D.\xD0\xBD.\xD1\x8D.") })
                });
                static BS_CYRL_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xD0\xBF. \xD0\xBD. \xD0\xB5.\xD0\xBD. \xD0\xB5.") })
                });
                static DOI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xE0\xA4\x88.\xE0\xA4\xAA\xE0\xA5\x82.\xE0\xA4\x88. \xE0\xA4\xB8\xE0\xA4\xA8\xE0\xA5\x8D") })
                });
                static DOI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xE0\xA4\x88.\xE0\xA4\xAA\xE0\xA5\x82.\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA5\x80") })
                });
                static PA_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0\xE0\xA8\x88.\xE0\xA8\xAA\xE0\xA9\x82.\xE0\xA8\xB8\xE0\xA9\xB0\xE0\xA8\xA8") })
                });
                static BR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0a-raok J.K.goude J.K.") })
                });
                static IS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0B\0fyrir Kristeftir Krist") })
                });
                static SO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0Ciise HortiiCiise Dabadii") })
                });
                static GD_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0Ro Chr\xC3\xACostaAn d\xC3\xA8idh Chr\xC3\xACosta") })
                });
                static UK_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xD0\xB4\xD0\xBE \xD0\xBD. \xD0\xB5.\xD0\xBD. \xD0\xB5.") })
                });
                static RU_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xD0\xB4\xD0\xBE \xD0\xBD. \xD1\x8D.\xD0\xBD. \xD1\x8D.") })
                });
                static HE_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xD7\x9C\xD7\xA4\xD7\xA0\xD7\x94\xD7\xB4\xD7\xA1\xD7\x9C\xD7\xA1\xD7\xA4\xD7\x99\xD7\xA8\xD7\x94") })
                });
                static SD_DEVA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xE0\xA4\xAC\xE0\xA5\x80\xE0\xA4\xB8\xE0\xA5\x80\xE0\xA4\x8F\xE0\xA4\xA1\xE0\xA5\x80") })
                });
                static PA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xE0\xA8\x88. \xE0\xA8\xAA\xE0\xA9\x82.\xE0\xA8\xB8\xE0\xA9\xB0\xE0\xA8\xA8") })
                });
                static MY_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xE1\x80\x98\xE1\x80\xAE\xE1\x80\x85\xE1\x80\xAE\xE1\x80\xA1\xE1\x80\x92\xE1\x80\xB1\xE1\x80\xAE") })
                });
                static KA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xE1\x83\xAB\xE1\x83\x95. \xE1\x83\xAC.\xE1\x83\x90\xE1\x83\xAE. \xE1\x83\xAC.") })
                });
                static FF_ADLM_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0\xF0\x9E\xA4\x80\xF0\x9E\xA4\x80\xF0\x9E\xA4\x8B\xF0\x9E\xA4\x87\xF0\x9E\xA4\x80\xF0\x9E\xA4\x8B") })
                });
                static IA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0ante Christopost Christo") })
                });
                static DA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0f\xC3\xB8r Kristusefter Kristus") })
                });
                static NB_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0f\xC3\xB8r Kristusetter Kristus") })
                });
                static SR_LATN_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0pre nove erenove ere") })
                });
                static SK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0pred Kristompo Kristovi") })
                });
                static HR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0C\0prije Kristaposlije Krista") })
                });
                static TR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0Milattan \xC3\x96nceMilattan Sonra") })
                });
                static GA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0Roimh Chr\xC3\xADostAnno Domini") })
                });
                static ID_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0Sebelum MasehiMasehi") })
                });
                static BRX_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0\xE0\xA4\xAC\xE0\xA4\xBF.\xE0\xA4\xB8\xE0\xA4\xBF.\xE0\xA4\x8F.\xE0\xA4\xA6\xE0\xA4\xBF") })
                });
                static GU_X_4: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0\xE0\xAA\x87 \xE0\xAA\xB8 \xE0\xAA\xAA\xE0\xAB\x81\xE0\xAA\x87\xE0\xAA\xB8") })
                });
                static TA_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0\xE0\xAE\x95\xE0\xAE\xBF.\xE0\xAE\xAE\xE0\xAF\x81.\xE0\xAE\x95\xE0\xAE\xBF.\xE0\xAE\xAA\xE0\xAE\xBF.") })
                });
                static CA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0abans de Cristdespr\xC3\xA9s de Crist") })
                });
                static SL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0pred Kristusompo Kristusu") })
                });
                static LT_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0prie\xC5\xA1 Krist\xC5\xB3po Kristaus") })
                });
                static BS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0E\0prije nove erenove ere") })
                });
                static PCM_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0Bif\xE1\xBB\x8D\xCC\x81 KraistKraist Im Yi\xE1\xBA\xB9") })
                });
                static SW_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0Kabla ya KristoBaada ya Kristo") })
                });
                static HU_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0Krisztus el\xC5\x91ttid\xC5\x91sz\xC3\xA1m\xC3\xADt\xC3\xA1sunk szerint") })
                });
                static SD_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0\xD9\x82\xD8\xA8\xD9\x84 \xD9\x85\xD8\xB3\xD9\x8A\xD8\xAD\xD8\xB9\xD9\x8A\xD8\xB3\xD9\x88\xD9\x8A \xDA\xA9\xD8\xA7\xD9\x86 \xD9\xBE\xD9\x87\xD8\xB1\xD9\x8A\xD9\x86") })
                });
                static UR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0\xD9\x82\xD8\xA8\xD9\x84 \xD9\x85\xD8\xB3\xDB\x8C\xD8\xAD\xD8\xB9\xDB\x8C\xD8\xB3\xD9\x88\xDB\x8C") })
                });
                static PT_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0antes de Cristodepois de Cristo") })
                });
                static GL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0antes de Cristodespois de Cristo") })
                });
                static ES_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0antes de Cristodespu\xC3\xA9s de Cristo") })
                });
                static KEA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x0F\0antis di Kristudispos di Kristu") })
                });
                static YRL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x10\0Kiristu sen\xC5\xA9d\xC3\xA9Kiristu arir\xC3\xA9") })
                });
                static UZ_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x10\0miloddan avvalgimilodiy") })
                });
                static JV_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x11\0Sakdurunge MasehiMasehi") })
                });
                static MR_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x11\0\xE0\xA4\x88. \xE0\xA4\xB8. \xE0\xA4\xAA\xE0\xA5\x82.\xE0\xA4\x87. \xE0\xA4\xB8.") })
                });
                static AST_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x11\0enantes de Cristudespu\xC3\xA9s de Cristu") })
                });
                static LV_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x11\0pirms m\xC5\xABsu \xC4\x93rasm\xC5\xABsu \xC4\x93r\xC4\x81") })
                });
                static PL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x11\0przed nasz\xC4\x85 er\xC4\x85naszej ery") })
                });
                static TE_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x12\0\xE0\xB0\x95\xE0\xB1\x8D\xE0\xB0\xB0\xE0\xB1\x80\xE0\xB0\xAA\xE0\xB1\x82\xE0\xB0\x95\xE0\xB1\x8D\xE0\xB0\xB0\xE0\xB1\x80\xE0\xB0\xB6") })
                });
                static KM_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x12\0\xE1\x9E\x98\xE1\x9E\xBB\xE1\x9E\x93 \xE1\x9E\x82.\xE1\x9E\x9F.\xE1\x9E\x82.\xE1\x9E\x9F.") })
                });
                static BR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x12\0a-raok Jezuz-Kristgoude Jezuz-Krist") })
                });
                static AZ_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x12\0eram\xC4\xB1zdan \xC9\x99vv\xC9\x99lyeni era") })
                });
                static SC_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x12\0in antis de Cristua pustis de Cristu") })
                });
                static RO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x13\0\xC3\xAEnainte de Hristosdup\xC4\x83 Hristos") })
                });
                static KS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x13\0\xD9\x82\xD8\xA8\xD9\x95\xD9\x84 \xD9\x85\xD8\xB3\xDB\x8C\xD9\x96\xD8\xAD\xD8\xA7\xDB\x8C\xD9\x86\xD9\x88 \xDA\x88\xD9\x88\xD9\x85\xD9\x86\xDB\x8C") })
                });
                static KN_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x13\0\xE0\xB2\x95\xE0\xB3\x8D\xE0\xB2\xB0\xE0\xB2\xBF.\xE0\xB2\xAA\xE0\xB3\x82\xE0\xB2\x95\xE0\xB3\x8D\xE0\xB2\xB0\xE0\xB2\xBF.\xE0\xB2\xB6") })
                });
                static AM_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x13\0\xE1\x8B\x93\xE1\x88\x98\xE1\x89\xB0 \xE1\x8B\x93\xE1\x88\x88\xE1\x88\x9D\xE1\x8B\x93\xE1\x88\x98\xE1\x89\xB0 \xE1\x88\x9D\xE1\x88\x95\xE1\x88\xA8\xE1\x89\xB5") })
                });
                static FR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x13\0avant J\xC3\xA9sus-Christapr\xC3\xA8s J\xC3\xA9sus-Christ") })
                });
                static HA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x14\0Kafin haihuwar annabBayan haihuwar annab") })
                });
                static CEB_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x14\0Sa Wala Pa Si KristoAnno Domini") })
                });
                static MNI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x14\0\xE0\xA6\x96\xE0\xA7\x83: \xE0\xA6\xAE\xE0\xA6\xAE\xE0\xA6\xBE\xE0\xA6\x82\xE0\xA6\x96\xE0\xA7\x83: \xE0\xA6\xAE\xE0\xA6\xA4\xE0\xA7\x81\xE0\xA6\x82") })
                });
                static ML_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x14\0\xE0\xB4\x95\xE0\xB5\x8D\xE0\xB4\xB0\xE0\xB4\xBF.\xE0\xB4\xAE\xE0\xB5\x81.\xE0\xB4\x8E\xE0\xB4\xA1\xE0\xB4\xBF") })
                });
                static VI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0Tr\xC6\xB0\xE1\xBB\x9Bc Thi\xC3\xAAn Ch\xC3\xBAaSau C\xC3\xB4ng Nguy\xC3\xAAn") })
                });
                static EL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0\xCF\x80\xCF\x81\xCE\xBF \xCE\xA7\xCF\x81\xCE\xB9\xCF\x83\xCF\x84\xCE\xBF\xCF\x8D\xCE\xBC\xCE\xB5\xCF\x84\xCE\xAC \xCE\xA7\xCF\x81\xCE\xB9\xCF\x83\xCF\x84\xCF\x8C\xCE\xBD") })
                });
                static HE_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0\xD7\x9C\xD7\xA4\xD7\xA0\xD7\x99 \xD7\x94\xD7\xA1\xD7\xA4\xD7\x99\xD7\xA8\xD7\x94\xD7\x9C\xD7\xA1\xD7\xA4\xD7\x99\xD7\xA8\xD7\x94") })
                });
                static AR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0\xD9\x82\xD8\xA8\xD9\x84 \xD8\xA7\xD9\x84\xD9\x85\xD9\x8A\xD9\x84\xD8\xA7\xD8\xAF\xD9\x85\xD9\x8A\xD9\x84\xD8\xA7\xD8\xAF\xD9\x8A") })
                });
                static TH_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0\xE0\xB8\x81\xE0\xB9\x88\xE0\xB8\xAD\xE0\xB8\x99 \xE0\xB8\x84.\xE0\xB8\xA8.\xE0\xB8\x84.\xE0\xB8\xA8.") })
                });
                static LO_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x15\0\xE0\xBA\x81\xE0\xBB\x88\xE0\xBA\xAD\xE0\xBA\x99 \xE0\xBA\x84.\xE0\xBA\xAA.\xE0\xBA\x84.\xE0\xBA\xAA.") })
                });
                static TG_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x16\0\xD0\x9F\xD0\xB5\xD1\x88 \xD0\xB0\xD0\xB7 \xD0\xBC\xD0\xB8\xD0\xBB\xD0\xBE\xD0\xB4\xD0\x9F\xD0\xB0\xD1\x81 \xD0\xB0\xD0\xB7 \xD0\xBC\xD0\xB8\xD0\xBB\xD0\xBE\xD0\xB4") })
                });
                static UK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x16\0\xD0\xB4\xD0\xBE \xD0\xBD\xD0\xB0\xD1\x88\xD0\xBE\xD1\x97 \xD0\xB5\xD1\x80\xD0\xB8\xD0\xBD\xD0\xB0\xD1\x88\xD0\xBE\xD1\x97 \xD0\xB5\xD1\x80\xD0\xB8") })
                });
                static SR_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x16\0\xD0\xBF\xD1\x80\xD0\xB5 \xD0\xBD\xD0\xBE\xD0\xB2\xD0\xB5 \xD0\xB5\xD1\x80\xD0\xB5\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xB5 \xD0\xB5\xD1\x80\xD0\xB5") })
                });
                static FA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x16\0\xD9\x82\xD8\xA8\xD9\x84 \xD8\xA7\xD8\xB2 \xD9\x85\xDB\x8C\xD9\x84\xD8\xA7\xD8\xAF\xD9\x85\xDB\x8C\xD9\x84\xD8\xA7\xD8\xAF\xDB\x8C") })
                });
                static BG_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x17\0\xD0\xBF\xD1\x80\xD0\xB5\xD0\xB4\xD0\xB8 \xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xB0\xD1\x81\xD0\xBB\xD0\xB5\xD0\xB4 \xD0\xA5\xD1\x80\xD0\xB8\xD1\x81\xD1\x82\xD0\xB0") })
                });
                static SI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x17\0\xE0\xB6\x9A\xE0\xB7\x8A\xE2\x80\x8D\xE0\xB6\xBB\xE0\xB7\x92.\xE0\xB6\xB4\xE0\xB7\x96.\xE0\xB6\x9A\xE0\xB7\x8A\xE2\x80\x8D\xE0\xB6\xBB\xE0\xB7\x92.\xE0\xB7\x80.") })
                });
                static BN_IN_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x18\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\x83\xE0\xA6\xAA\xE0\xA7\x82\xE0\xA6\x83\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA6\xB0\xE0\xA6\xBF\xE0\xA6\x83") })
                });
                static NE_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xBE \xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\xB8\xE0\xA4\xA8\xE0\xA5\x8D") })
                });
                static HI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xBE-\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA5\x80 \xE0\xA4\xB8\xE0\xA4\xA8") })
                });
                static MAI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xBE-\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xB5\xE0\xA5\x80") })
                });
                static HI_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xBE-\xE0\xA4\xAA\xE0\xA5\x82\xE0\xA4\xB0\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA5\x80") })
                });
                static AS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA7\xB0\xE0\xA7\x80\xE0\xA6\x83 \xE0\xA6\xAA\xE0\xA7\x82\xE0\xA6\x83\xE0\xA6\x96\xE0\xA7\x8D\xE0\xA7\xB0\xE0\xA7\x80\xE0\xA6\x83") })
                });
                static PA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE0\xA8\x88\xE0\xA8\xB8\xE0\xA8\xB5\xE0\xA9\x80 \xE0\xA8\xAA\xE0\xA9\x82\xE0\xA8\xB0\xE0\xA8\xB5\xE0\xA8\x88\xE0\xA8\xB8\xE0\xA8\xB5\xE0\xA9\x80 \xE0\xA8\xB8\xE0\xA9\xB0\xE0\xA8\xA8") })
                });
                static TI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0\xE1\x89\x85\xE1\x8B\xB5\xE1\x88\x98 \xE1\x8A\xAD\xE1\x88\xAD\xE1\x88\xB5\xE1\x89\xB6\xE1\x88\xB5\xE1\x8B\x93\xE1\x88\x98\xE1\x89\xB0 \xE1\x88\x9D\xE1\x88\x95\xE1\x88\xA8\xE1\x89\xB5") })
                });
                static CS_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x19\0p\xC5\x99ed na\xC5\xA1\xC3\xADm letopo\xC4\x8Dtemna\xC5\xA1eho letopo\xC4\x8Dtu") })
                });
                static BS_CYRL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1A\0\xD0\xBF\xD1\x80\xD0\xB8\xD1\x98\xD0\xB5 \xD0\xBD\xD0\xBE\xD0\xB2\xD0\xB5 \xD0\xB5\xD1\x80\xD0\xB5\xD0\xBD\xD0\xBE\xD0\xB2\xD0\xB5 \xD0\xB5\xD1\x80\xD0\xB5") })
                });
                static GU_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1A\0\xE0\xAA\x88.\xE0\xAA\xB8.\xE0\xAA\xAA\xE0\xAB\x82\xE0\xAA\xB0\xE0\xAB\x8D\xE0\xAA\xB5\xE0\xAB\x87\xE0\xAA\x88.\xE0\xAA\xB8.") })
                });
                static HY_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1B\0\xD5\x94\xD6\x80\xD5\xAB\xD5\xBD\xD5\xBF\xD5\xB8\xD5\xBD\xD5\xAB\xD6\x81 \xD5\xA1\xD5\xBC\xD5\xA1\xD5\xBB\xD5\x94\xD6\x80\xD5\xAB\xD5\xBD\xD5\xBF\xD5\xB8\xD5\xBD\xD5\xAB\xD6\x81 \xD5\xB0\xD5\xA5\xD5\xBF\xD5\xB8") })
                });
                static FI_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1B\0ennen Kristuksen syntym\xC3\xA4\xC3\xA4j\xC3\xA4lkeen Kristuksen syntym\xC3\xA4n") })
                });
                static MK_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1C\0\xD0\xBF\xD1\x80\xD0\xB5\xD0\xB4 \xD0\xBD\xD0\xB0\xD1\x88\xD0\xB0\xD1\x82\xD0\xB0 \xD0\xB5\xD1\x80\xD0\xB0\xD0\xBE\xD0\xB4 \xD0\xBD\xD0\xB0\xD1\x88\xD0\xB0\xD1\x82\xD0\xB0 \xD0\xB5\xD1\x80\xD0\xB0") })
                });
                static PS_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1C\0\xD9\x84\xD9\x87 \xD9\x85\xDB\x8C\xD9\x84\xD8\xA7\xD8\xAF \xD9\x88\xDA\x93\xD8\xA7\xD9\x86\xD8\xAF\xDB\x90\xD9\x85.") })
                });
                static KS_DEVA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1C\0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA4\xBE \xE0\xA4\xAC\xE0\xA5\x8D\xE0\xA4\xB0\xE0\xA5\x8B\xE0\xA4\x82\xE0\xA4\xA0\xE0\xA4\x88\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xB5\xE0\xA5\x80") })
                });
                static SAT_X_3: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1C\0\xE1\xB1\xA5\xE1\xB1\xAE\xE1\xB1\xA8\xE1\xB1\xA2\xE1\xB1\x9F \xE1\xB1\x9E\xE1\xB1\x9F\xE1\xB1\xA6\xE1\xB1\x9F\xE1\xB1\xA4\xE1\xB1\xA5\xE1\xB1\xA3\xE1\xB1\xA4") })
                });
                static DSB_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1C\0p\xC5\x9Bed Kristusowym naro\xC5\xBAenimp\xC3\xB3 Kristusowem naro\xC5\xBAenju") })
                });
                static HSB_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1D\0p\xC5\x99ed Chrystowym narod\xC5\xBAenjompo Chrystowym narod\xC5\xBAenju") })
                });
                static UZ_CYRL_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x1F\0\xD0\xBC\xD0\xB8\xD0\xBB\xD0\xBE\xD0\xB4\xD0\xB4\xD0\xB0\xD0\xBD \xD0\xB0\xD0\xB2\xD0\xB2\xD0\xB0\xD0\xBB\xD0\xB3\xD0\xB8\xD0\xBC\xD0\xB8\xD0\xBB\xD0\xBE\xD0\xB4\xD0\xB8\xD0\xB9") })
                });
                static LO_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\x000\0\xE0\xBA\x81\xE0\xBB\x88\xE0\xBA\xAD\xE0\xBA\x99\xE0\xBA\x84\xE0\xBA\xA3\xE0\xBA\xB4\xE0\xBA\x94\xE0\xBA\xAA\xE0\xBA\xB1\xE0\xBA\x81\xE0\xBA\x81\xE0\xBA\xB0\xE0\xBA\xA5\xE0\xBA\xB2\xE0\xBA\x94\xE0\xBA\x84\xE0\xBA\xA3\xE0\xBA\xB4\xE0\xBA\x94\xE0\xBA\xAA\xE0\xBA\xB1\xE0\xBA\x81\xE0\xBA\x81\xE0\xBA\xB0\xE0\xBA\xA5\xE0\xBA\xB2\xE0\xBA\x94") })
                });
                static KM_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\x000\0\xE1\x9E\x98\xE1\x9E\xBB\xE1\x9E\x93\xE2\x80\x8B\xE1\x9E\x82\xE1\x9F\x92\xE1\x9E\x9A\xE1\x9E\xB7\xE1\x9E\x9F\xE1\x9F\x92\xE1\x9E\x8F\xE1\x9E\x9F\xE1\x9E\x80\xE1\x9E\x9A\xE1\x9E\xB6\xE1\x9E\x87\xE1\x9E\x82\xE1\x9F\x92\xE1\x9E\x9A\xE1\x9E\xB7\xE1\x9E\x9F\xE1\x9F\x92\xE1\x9E\x8F\xE1\x9E\x9F\xE1\x9E\x80\xE1\x9E\x9A\xE1\x9E\xB6\xE1\x9E\x87") })
                });
                static TA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\x007\0\xE0\xAE\x95\xE0\xAE\xBF\xE0\xAE\xB1\xE0\xAE\xBF\xE0\xAE\xB8\xE0\xAF\x8D\xE0\xAE\xA4\xE0\xAF\x81\xE0\xAE\xB5\xE0\xAF\x81\xE0\xAE\x95\xE0\xAF\x8D\xE0\xAE\x95\xE0\xAF\x81 \xE0\xAE\xAE\xE0\xAF\x81\xE0\xAE\xA9\xE0\xAF\x8D\xE0\xAE\x85\xE0\xAE\xA9\xE0\xAF\x8D\xE0\xAE\xA9\xE0\xAF\x8B \xE0\xAE\x9F\xE0\xAF\x8B\xE0\xAE\xAE\xE0\xAE\xBF\xE0\xAE\xA9\xE0\xAE\xBF") })
                });
                static KA_X_5: <icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable = icu::datetime::provider::neo::YearSymbolsV1::Eras(unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x03\0bcece") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\x007\0\xE1\x83\xAB\xE1\x83\x95\xE1\x83\x94\xE1\x83\x9A\xE1\x83\x98 \xE1\x83\xAC\xE1\x83\x94\xE1\x83\x9A\xE1\x83\x97\xE1\x83\x90\xE1\x83\xA6\xE1\x83\xA0\xE1\x83\x98\xE1\x83\xAA\xE1\x83\xAE\xE1\x83\x95\xE1\x83\x98\xE1\x83\x97\xE1\x83\x90\xE1\x83\xAE\xE1\x83\x90\xE1\x83\x9A\xE1\x83\x98 \xE1\x83\xAC\xE1\x83\x94\xE1\x83\x9A\xE1\x83\x97\xE1\x83\x90\xE1\x83\xA6\xE1\x83\xA0\xE1\x83\x98\xE1\x83\xAA\xE1\x83\xAE\xE1\x83\x95\xE1\x83\x98\xE1\x83\x97") })
                });
                static VALUES: [&<icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::DataMarker>::Yokeable; 444usize] = [&AF_X_3, &AF_X_3, &AF_X_5, &AM_X_3, &AM_X_3, &AM_X_5, &AR_X_3, &AR_X_3, &AR_X_5, &AS_X_3, &AS_X_3, &AS_X_5, &AST_X_3, &AST_X_3, &AST_X_5, &AZ_X_3, &AZ_X_3, &AZ_X_5, &BE_X_3, &BE_X_3, &BE_X_5, &BG_X_3, &BG_X_3, &BG_X_5, &BN_IN_X_3, &BN_IN_X_3, &BN_IN_X_5, &BN_X_3, &BN_X_3, &BN_X_5, &BR_X_3, &BR_X_3, &BR_X_5, &BRX_X_3, &BRX_X_3, &BRX_X_5, &BS_CYRL_X_3, &BS_CYRL_X_4, &BS_CYRL_X_5, &BS_X_3, &BS_X_4, &BS_X_5, &CA_X_3, &CA_X_3, &CA_X_5, &CEB_X_3, &CEB_X_3, &CEB_X_5, &CEB_X_3, &CEB_X_3, &CHR_X_5, &CS_X_3, &CS_X_4, &CS_X_5, &CV_X_3, &CV_X_3, &CV_X_5, &CY_X_3, &CY_X_4, &CY_X_5, &DA_X_3, &DA_X_4, &DA_X_5, &DE_X_3, &DE_X_3, &DE_X_3, &DOI_X_3, &DOI_X_3, &DOI_X_5, &DSB_X_3, &DSB_X_3, &DSB_X_5, &EL_X_3, &EL_X_3, &EL_X_5, &EN_CA_X_5, &CEB_X_3, &EN_X_4, &EN_X_5, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_419_X_3, &ES_X_3, &ES_X_3, &ES_X_5, &ET_X_3, &ET_X_3, &ET_X_5, &EU_X_3, &EU_X_4, &EU_X_5, &FA_X_3, &FA_X_4, &FA_X_5, &FF_ADLM_X_3, &FF_ADLM_X_3, &FF_ADLM_X_5, &FI_X_3, &FI_X_4, &FI_X_5, &CEB_X_3, &CEB_X_3, &EN_X_5, &DA_X_3, &DA_X_4, &FO_X_5, &FR_X_3, &FR_X_3, &FR_X_5, &GA_X_3, &GA_X_3, &GA_X_5, &GA_X_3, &GD_X_4, &GD_X_5, &ES_419_X_3, &ES_419_X_3, &GL_X_5, &GU_X_3, &GU_X_4, &GU_X_5, &HA_X_3, &HA_X_3, &HA_X_5, &HE_X_3, &HE_X_4, &HE_X_5, &CEB_X_3, &EN_X_4, &EN_X_5, &HI_X_3, &HI_X_3, &HI_X_5, &HR_X_3, &HR_X_4, &HR_X_5, &HSB_X_3, &HSB_X_3, &HSB_X_5, &HU_X_3, &HU_X_4, &HU_X_5, &HY_X_3, &HY_X_3, &HY_X_5, &IA_X_3, &IA_X_3, &IA_X_5, &ID_X_3, &ID_X_3, &ID_X_5, &IG_X_3, &IG_X_3, &IG_X_5, &DA_X_3, &IS_X_4, &IS_X_5, &ES_419_X_3, &CA_X_3, &IT_X_5, &JA_X_3, &CEB_X_3, &JA_X_3, &ID_X_3, &ID_X_3, &JV_X_5, &KA_X_3, &KA_X_3, &KA_X_5, &KEA_X_3, &KEA_X_3, &KEA_X_5, &KGP_X_3, &KGP_X_3, &KGP_X_5, &KK_X_3, &KK_X_3, &KK_X_5, &KM_X_3, &KM_X_3, &KM_X_5, &KN_X_3, &KN_X_3, &KN_X_5, &CEB_X_3, &CEB_X_3, &KO_X_5, &KOK_X_3, &KOK_X_3, &KOK_X_5, &CEB_X_3, &CEB_X_3, &KS_DEVA_X_5, &KS_X_3, &KS_X_3, &KS_X_5, &KY_X_3, &KY_X_3, &KY_X_5, &LO_X_3, &LO_X_3, &LO_X_5, &LT_X_3, &LT_X_3, &LT_X_5, &LV_X_3, &LV_X_3, &LV_X_5, &MAI_X_3, &MAI_X_3, &MAI_X_3, &BS_CYRL_X_4, &BS_CYRL_X_4, &MK_X_5, &ML_X_3, &ML_X_3, &ML_X_5, &MN_X_3, &MN_X_3, &MN_X_5, &MNI_X_3, &MNI_X_3, &MNI_X_3, &MR_X_3, &MR_X_3, &MR_X_5, &MS_X_3, &MS_X_3, &MS_X_3, &MY_X_3, &MY_X_3, &MY_X_5, &DA_X_3, &DA_X_3, &NB_X_5, &NE_X_3, &NE_X_3, &NE_X_3, &NL_X_3, &AF_X_3, &AF_X_5, &DA_X_3, &DA_X_3, &NB_X_5, &DA_X_3, &DA_X_3, &NB_X_5, &CEB_X_3, &CEB_X_3, &OR_X_5, &PA_X_3, &PA_X_4, &PA_X_5, &PCM_X_3, &PCM_X_3, &PCM_X_5, &PL_X_3, &PL_X_3, &PL_X_5, &PS_X_3, &PS_X_3, &PS_X_5, &ES_419_X_3, &ES_419_X_3, &PT_X_5, &QU_X_3, &QU_X_4, &QU_X_5, &RM_X_3, &RM_X_3, &RM_X_5, &RO_X_3, &RO_X_3, &RO_X_5, &RU_X_3, &RU_X_4, &RU_X_5, &SAT_X_3, &SAT_X_3, &SAT_X_3, &SC_X_3, &SC_X_3, &SC_X_5, &SD_DEVA_X_3, &SD_DEVA_X_3, &SD_DEVA_X_3, &SD_X_3, &SD_X_3, &SD_X_5, &SI_X_3, &SI_X_3, &SI_X_5, &SK_X_3, &SK_X_3, &SK_X_5, &LT_X_3, &LT_X_3, &SL_X_5, &CEB_X_3, &EN_X_4, &SO_X_5, &SQ_X_3, &SQ_X_3, &SQ_X_5, &BS_CYRL_X_5, &BS_X_5, &BS_X_3, &PL_X_3, &SR_LATN_X_5, &BS_X_5, &BS_CYRL_X_3, &BS_CYRL_X_4, &SR_X_5, &ID_X_3, &ID_X_3, &ID_X_3, &DA_X_3, &DA_X_3, &SV_X_5, &SW_X_3, &SW_X_3, &SW_X_5, &TA_X_3, &TA_X_3, &TA_X_5, &TE_X_3, &TE_X_3, &TE_X_5, &TG_X_3, &TG_X_3, &TG_X_5, &TH_X_3, &TH_X_3, &TH_X_5, &AM_X_5, &AM_X_3, &AM_X_3, &TI_X_5, &TK_X_3, &TK_X_3, &TK_X_5, &TO_X_3, &TO_X_3, &TO_X_5, &TR_X_3, &TR_X_3, &TR_X_5, &TT_X_3, &TT_X_3, &TT_X_5, &UK_X_3, &UK_X_4, &UK_X_5, &UND_X_3, &UND_X_3, &UND_X_3, &UR_X_3, &UR_X_3, &UR_X_3, &UZ_CYRL_X_3, &UZ_CYRL_X_3, &UZ_CYRL_X_5, &UZ_X_3, &UZ_X_3, &UZ_X_5, &VI_X_3, &VI_X_4, &VI_X_5, &WO_X_3, &WO_X_3, &WO_X_5, &CEB_X_3, &CEB_X_3, &CEB_X_3, &YO_X_3, &YO_X_3, &YO_X_5, &YRL_X_3, &YRL_X_3, &YRL_X_5, &YUE_HANS_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &YUE_HANS_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &ZH_HK_X_3, &CEB_X_3, &CEB_X_3, &CEB_X_3];
                static KEYS: [&str; 444usize] = ["af-x-3", "af-x-4", "af-x-5", "am-x-3", "am-x-4", "am-x-5", "ar-x-3", "ar-x-4", "ar-x-5", "as-x-3", "as-x-4", "as-x-5", "ast-x-3", "ast-x-4", "ast-x-5", "az-x-3", "az-x-4", "az-x-5", "be-x-3", "be-x-4", "be-x-5", "bg-x-3", "bg-x-4", "bg-x-5", "bn-IN-x-3", "bn-IN-x-4", "bn-IN-x-5", "bn-x-3", "bn-x-4", "bn-x-5", "br-x-3", "br-x-4", "br-x-5", "brx-x-3", "brx-x-4", "brx-x-5", "bs-Cyrl-x-3", "bs-Cyrl-x-4", "bs-Cyrl-x-5", "bs-x-3", "bs-x-4", "bs-x-5", "ca-x-3", "ca-x-4", "ca-x-5", "ceb-x-3", "ceb-x-4", "ceb-x-5", "chr-x-3", "chr-x-4", "chr-x-5", "cs-x-3", "cs-x-4", "cs-x-5", "cv-x-3", "cv-x-4", "cv-x-5", "cy-x-3", "cy-x-4", "cy-x-5", "da-x-3", "da-x-4", "da-x-5", "de-x-3", "de-x-4", "de-x-5", "doi-x-3", "doi-x-4", "doi-x-5", "dsb-x-3", "dsb-x-4", "dsb-x-5", "el-x-3", "el-x-4", "el-x-5", "en-CA-x-5", "en-x-3", "en-x-4", "en-x-5", "es-419-x-3", "es-419-x-4", "es-AR-x-3", "es-AR-x-4", "es-BO-x-3", "es-BO-x-4", "es-BR-x-3", "es-BR-x-4", "es-BZ-x-3", "es-BZ-x-4", "es-CL-x-3", "es-CL-x-4", "es-CO-x-3", "es-CO-x-4", "es-CR-x-3", "es-CR-x-4", "es-CU-x-3", "es-CU-x-4", "es-DO-x-3", "es-DO-x-4", "es-EC-x-3", "es-EC-x-4", "es-GT-x-3", "es-GT-x-4", "es-HN-x-3", "es-HN-x-4", "es-MX-x-3", "es-MX-x-4", "es-NI-x-3", "es-NI-x-4", "es-PA-x-3", "es-PA-x-4", "es-PE-x-3", "es-PE-x-4", "es-PR-x-3", "es-PR-x-4", "es-PY-x-3", "es-PY-x-4", "es-SV-x-3", "es-SV-x-4", "es-US-x-3", "es-US-x-4", "es-UY-x-3", "es-UY-x-4", "es-VE-x-3", "es-VE-x-4", "es-x-3", "es-x-4", "es-x-5", "et-x-3", "et-x-4", "et-x-5", "eu-x-3", "eu-x-4", "eu-x-5", "fa-x-3", "fa-x-4", "fa-x-5", "ff-Adlm-x-3", "ff-Adlm-x-4", "ff-Adlm-x-5", "fi-x-3", "fi-x-4", "fi-x-5", "fil-x-3", "fil-x-4", "fil-x-5", "fo-x-3", "fo-x-4", "fo-x-5", "fr-x-3", "fr-x-4", "fr-x-5", "ga-x-3", "ga-x-4", "ga-x-5", "gd-x-3", "gd-x-4", "gd-x-5", "gl-x-3", "gl-x-4", "gl-x-5", "gu-x-3", "gu-x-4", "gu-x-5", "ha-x-3", "ha-x-4", "ha-x-5", "he-x-3", "he-x-4", "he-x-5", "hi-Latn-x-3", "hi-Latn-x-4", "hi-Latn-x-5", "hi-x-3", "hi-x-4", "hi-x-5", "hr-x-3", "hr-x-4", "hr-x-5", "hsb-x-3", "hsb-x-4", "hsb-x-5", "hu-x-3", "hu-x-4", "hu-x-5", "hy-x-3", "hy-x-4", "hy-x-5", "ia-x-3", "ia-x-4", "ia-x-5", "id-x-3", "id-x-4", "id-x-5", "ig-x-3", "ig-x-4", "ig-x-5", "is-x-3", "is-x-4", "is-x-5", "it-x-3", "it-x-4", "it-x-5", "ja-x-3", "ja-x-4", "ja-x-5", "jv-x-3", "jv-x-4", "jv-x-5", "ka-x-3", "ka-x-4", "ka-x-5", "kea-x-3", "kea-x-4", "kea-x-5", "kgp-x-3", "kgp-x-4", "kgp-x-5", "kk-x-3", "kk-x-4", "kk-x-5", "km-x-3", "km-x-4", "km-x-5", "kn-x-3", "kn-x-4", "kn-x-5", "ko-x-3", "ko-x-4", "ko-x-5", "kok-x-3", "kok-x-4", "kok-x-5", "ks-Deva-x-3", "ks-Deva-x-4", "ks-Deva-x-5", "ks-x-3", "ks-x-4", "ks-x-5", "ky-x-3", "ky-x-4", "ky-x-5", "lo-x-3", "lo-x-4", "lo-x-5", "lt-x-3", "lt-x-4", "lt-x-5", "lv-x-3", "lv-x-4", "lv-x-5", "mai-x-3", "mai-x-4", "mai-x-5", "mk-x-3", "mk-x-4", "mk-x-5", "ml-x-3", "ml-x-4", "ml-x-5", "mn-x-3", "mn-x-4", "mn-x-5", "mni-x-3", "mni-x-4", "mni-x-5", "mr-x-3", "mr-x-4", "mr-x-5", "ms-x-3", "ms-x-4", "ms-x-5", "my-x-3", "my-x-4", "my-x-5", "nb-x-3", "nb-x-4", "nb-x-5", "ne-x-3", "ne-x-4", "ne-x-5", "nl-x-3", "nl-x-4", "nl-x-5", "nn-x-3", "nn-x-4", "nn-x-5", "no-x-3", "no-x-4", "no-x-5", "or-x-3", "or-x-4", "or-x-5", "pa-x-3", "pa-x-4", "pa-x-5", "pcm-x-3", "pcm-x-4", "pcm-x-5", "pl-x-3", "pl-x-4", "pl-x-5", "ps-x-3", "ps-x-4", "ps-x-5", "pt-x-3", "pt-x-4", "pt-x-5", "qu-x-3", "qu-x-4", "qu-x-5", "rm-x-3", "rm-x-4", "rm-x-5", "ro-x-3", "ro-x-4", "ro-x-5", "ru-x-3", "ru-x-4", "ru-x-5", "sat-x-3", "sat-x-4", "sat-x-5", "sc-x-3", "sc-x-4", "sc-x-5", "sd-Deva-x-3", "sd-Deva-x-4", "sd-Deva-x-5", "sd-x-3", "sd-x-4", "sd-x-5", "si-x-3", "si-x-4", "si-x-5", "sk-x-3", "sk-x-4", "sk-x-5", "sl-x-3", "sl-x-4", "sl-x-5", "so-x-3", "so-x-4", "so-x-5", "sq-x-3", "sq-x-4", "sq-x-5", "sr-BA-x-5", "sr-Latn-BA-x-5", "sr-Latn-x-3", "sr-Latn-x-4", "sr-Latn-x-5", "sr-ME-x-5", "sr-x-3", "sr-x-4", "sr-x-5", "su-x-3", "su-x-4", "su-x-5", "sv-x-3", "sv-x-4", "sv-x-5", "sw-x-3", "sw-x-4", "sw-x-5", "ta-x-3", "ta-x-4", "ta-x-5", "te-x-3", "te-x-4", "te-x-5", "tg-x-3", "tg-x-4", "tg-x-5", "th-x-3", "th-x-4", "th-x-5", "ti-ER-x-5", "ti-x-3", "ti-x-4", "ti-x-5", "tk-x-3", "tk-x-4", "tk-x-5", "to-x-3", "to-x-4", "to-x-5", "tr-x-3", "tr-x-4", "tr-x-5", "tt-x-3", "tt-x-4", "tt-x-5", "uk-x-3", "uk-x-4", "uk-x-5", "und-x-3", "und-x-4", "und-x-5", "ur-x-3", "ur-x-4", "ur-x-5", "uz-Cyrl-x-3", "uz-Cyrl-x-4", "uz-Cyrl-x-5", "uz-x-3", "uz-x-4", "uz-x-5", "vi-x-3", "vi-x-4", "vi-x-5", "wo-x-3", "wo-x-4", "wo-x-5", "xh-x-3", "xh-x-4", "xh-x-5", "yo-x-3", "yo-x-4", "yo-x-5", "yrl-x-3", "yrl-x-4", "yrl-x-5", "yue-Hans-x-3", "yue-Hans-x-4", "yue-Hans-x-5", "yue-x-3", "yue-x-4", "yue-x-5", "zh-HK-x-3", "zh-HK-x-4", "zh-HK-x-5", "zh-Hant-x-3", "zh-Hant-x-4", "zh-Hant-x-5", "zh-MO-x-3", "zh-MO-x-4", "zh-MO-x-5", "zh-x-3", "zh-x-4", "zh-x-5", "zu-x-3", "zu-x-4", "zu-x-5"];
                let mut metadata = icu_provider::DataResponseMetadata::default();
                let payload = if let Ok(payload) = KEYS.binary_search_by(|k| req.locale.strict_cmp(k.as_bytes()).reverse()).map(|i| *unsafe { VALUES.get_unchecked(i) }) {
                    payload
                } else {
                    const FALLBACKER: icu::locid_transform::fallback::LocaleFallbackerWithConfig<'static> = icu::locid_transform::fallback::LocaleFallbacker::new().for_config(<icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::KeyedDataMarker>::KEY.fallback_config());
                    let mut fallback_iterator = FALLBACKER.fallback_for(req.locale.clone());
                    loop {
                        if fallback_iterator.get().is_und() {
                            return Err(icu_provider::DataErrorKind::MissingLocale.with_req(<icu::datetime::provider::neo::GregorianYearSymbolsV1Marker as icu_provider::KeyedDataMarker>::KEY, req));
                        }
                        if let Ok(payload) = KEYS.binary_search_by(|k| fallback_iterator.get().strict_cmp(k.as_bytes()).reverse()).map(|i| *unsafe { VALUES.get_unchecked(i) }) {
                            metadata.locale = Some(fallback_iterator.take());
                            break payload;
                        }
                        fallback_iterator.step();
                    }
                };
                Ok(icu_provider::DataResponse { payload: Some(icu_provider::DataPayload::from_static_ref(payload)), metadata })
            }
        }
    };
}