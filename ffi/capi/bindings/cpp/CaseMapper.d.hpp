#ifndef CaseMapper_D_HPP
#define CaseMapper_D_HPP

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <optional>
#include "diplomat_runtime.hpp"
#include "DataError.d.hpp"
#include "TitlecaseOptionsV1.d.hpp"

class CodePointSetBuilder;
class DataProvider;
class Locale;
struct TitlecaseOptionsV1;
class DataError;


namespace capi {
    typedef struct CaseMapper CaseMapper;
}

class CaseMapper {
public:

  inline static diplomat::result<std::unique_ptr<CaseMapper>, DataError> create(const DataProvider& provider);

  inline diplomat::result<std::string, diplomat::Utf8Error> lowercase(std::string_view s, const Locale& locale) const;

  inline diplomat::result<std::string, diplomat::Utf8Error> uppercase(std::string_view s, const Locale& locale) const;

  inline diplomat::result<std::string, diplomat::Utf8Error> titlecase_segment_with_only_case_data_v1(std::string_view s, const Locale& locale, TitlecaseOptionsV1 options) const;

  inline diplomat::result<std::string, diplomat::Utf8Error> fold(std::string_view s) const;

  inline diplomat::result<std::string, diplomat::Utf8Error> fold_turkic(std::string_view s) const;

  inline void add_case_closure_to(char32_t c, CodePointSetBuilder& builder) const;

  inline char32_t simple_lowercase(char32_t ch) const;

  inline char32_t simple_uppercase(char32_t ch) const;

  inline char32_t simple_titlecase(char32_t ch) const;

  inline char32_t simple_fold(char32_t ch) const;

  inline char32_t simple_fold_turkic(char32_t ch) const;

  inline const capi::CaseMapper* AsFFI() const;
  inline capi::CaseMapper* AsFFI();
  inline static const CaseMapper* FromFFI(const capi::CaseMapper* ptr);
  inline static CaseMapper* FromFFI(capi::CaseMapper* ptr);
  inline static void operator delete(void* ptr);
private:
  CaseMapper() = delete;
  CaseMapper(const CaseMapper&) = delete;
  CaseMapper(CaseMapper&&) noexcept = delete;
  CaseMapper operator=(const CaseMapper&) = delete;
  CaseMapper operator=(CaseMapper&&) noexcept = delete;
  static void operator delete[](void*, size_t) = delete;
};


#endif // CaseMapper_D_HPP