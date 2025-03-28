#ifndef icu4x_WeekCalculator_D_HPP
#define icu4x_WeekCalculator_D_HPP

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <functional>
#include <optional>
#include "../diplomat_runtime.hpp"

namespace icu4x {
namespace capi { struct DataProvider; }
class DataProvider;
namespace capi { struct Locale; }
class Locale;
namespace capi { struct WeekCalculator; }
class WeekCalculator;
struct WeekendContainsDay;
class DataError;
class Weekday;
}


namespace icu4x {
namespace capi {
    struct WeekCalculator;
} // namespace capi
} // namespace

namespace icu4x {
/**
 * A Week calculator, useful to be passed in to `week_of_year()` on Date and DateTime types
 *
 * See the [Rust documentation for `WeekCalculator`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html) for more information.
 */
class WeekCalculator {
public:

  /**
   * Creates a new [`WeekCalculator`] from locale data using compiled data.
   *
   * See the [Rust documentation for `try_new`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#method.try_new) for more information.
   */
  inline static diplomat::result<std::unique_ptr<icu4x::WeekCalculator>, icu4x::DataError> create(const icu4x::Locale& locale);

  /**
   * Creates a new [`WeekCalculator`] from locale data using a particular data source.
   *
   * See the [Rust documentation for `try_new`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#method.try_new) for more information.
   */
  inline static diplomat::result<std::unique_ptr<icu4x::WeekCalculator>, icu4x::DataError> create_with_provider(const icu4x::DataProvider& provider, const icu4x::Locale& locale);

  /**
   * Additional information: [1](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#structfield.first_weekday), [2](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#structfield.min_week_days)
   */
  inline static std::unique_ptr<icu4x::WeekCalculator> from_first_day_of_week_and_min_week_days(icu4x::Weekday first_weekday, uint8_t min_week_days);

  /**
   * Returns the weekday that starts the week for this object's locale
   *
   * See the [Rust documentation for `first_weekday`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#structfield.first_weekday) for more information.
   */
  inline icu4x::Weekday first_weekday() const;

  /**
   * The minimum number of days overlapping a year required for a week to be
   * considered part of that year
   *
   * See the [Rust documentation for `min_week_days`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#structfield.min_week_days) for more information.
   */
  inline uint8_t min_week_days() const;

  /**
   * See the [Rust documentation for `weekend`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#method.weekend) for more information.
   */
  inline icu4x::WeekendContainsDay weekend() const;

  inline const icu4x::capi::WeekCalculator* AsFFI() const;
  inline icu4x::capi::WeekCalculator* AsFFI();
  inline static const icu4x::WeekCalculator* FromFFI(const icu4x::capi::WeekCalculator* ptr);
  inline static icu4x::WeekCalculator* FromFFI(icu4x::capi::WeekCalculator* ptr);
  inline static void operator delete(void* ptr);
private:
  WeekCalculator() = delete;
  WeekCalculator(const icu4x::WeekCalculator&) = delete;
  WeekCalculator(icu4x::WeekCalculator&&) noexcept = delete;
  WeekCalculator operator=(const icu4x::WeekCalculator&) = delete;
  WeekCalculator operator=(icu4x::WeekCalculator&&) noexcept = delete;
  static void operator delete[](void*, size_t) = delete;
};

} // namespace
#endif // icu4x_WeekCalculator_D_HPP
