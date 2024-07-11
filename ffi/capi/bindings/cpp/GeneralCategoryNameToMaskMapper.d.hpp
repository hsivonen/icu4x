#ifndef GeneralCategoryNameToMaskMapper_D_HPP
#define GeneralCategoryNameToMaskMapper_D_HPP

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <optional>
#include "diplomat_runtime.hpp"
#include "DataError.d.hpp"

class DataProvider;
class DataError;


namespace capi {
    typedef struct GeneralCategoryNameToMaskMapper GeneralCategoryNameToMaskMapper;
}

class GeneralCategoryNameToMaskMapper {
public:

  inline uint32_t get_strict(std::string_view name) const;

  inline uint32_t get_loose(std::string_view name) const;

  inline static diplomat::result<std::unique_ptr<GeneralCategoryNameToMaskMapper>, DataError> load(const DataProvider& provider);

  inline const capi::GeneralCategoryNameToMaskMapper* AsFFI() const;
  inline capi::GeneralCategoryNameToMaskMapper* AsFFI();
  inline static const GeneralCategoryNameToMaskMapper* FromFFI(const capi::GeneralCategoryNameToMaskMapper* ptr);
  inline static GeneralCategoryNameToMaskMapper* FromFFI(capi::GeneralCategoryNameToMaskMapper* ptr);
  inline static void operator delete(void* ptr);
private:
  GeneralCategoryNameToMaskMapper() = delete;
  GeneralCategoryNameToMaskMapper(const GeneralCategoryNameToMaskMapper&) = delete;
  GeneralCategoryNameToMaskMapper(GeneralCategoryNameToMaskMapper&&) noexcept = delete;
  GeneralCategoryNameToMaskMapper operator=(const GeneralCategoryNameToMaskMapper&) = delete;
  GeneralCategoryNameToMaskMapper operator=(GeneralCategoryNameToMaskMapper&&) noexcept = delete;
  static void operator delete[](void*, size_t) = delete;
};


#endif // GeneralCategoryNameToMaskMapper_D_HPP