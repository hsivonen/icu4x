// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

//! This module contains types and implementations for the Persian calendar.
//!
//! ```rust
//! use icu::calendar::{Date, DateTime};
//!
//! // `Date` type
//! let persian_date = Date::try_new_persian_date(1348, 10, 11)
//!     .expect("Failed to initialize Persian Date instance.");
//!
//! // `DateTime` type
//! let persian_datetime = DateTime::try_new_persian_datetime(1348, 10, 11, 13, 1, 0)
//!     .expect("Failed to initialize Persian DateTime instance.");
//!
//! // `Date` checks
//! assert_eq!(persian_date.year().number, 1348);
//! assert_eq!(persian_date.month().ordinal, 10);
//! assert_eq!(persian_date.day_of_month().0, 11);
//!
//! // `DateTime` checks
//! assert_eq!(persian_datetime.date.year().number, 1348);
//! assert_eq!(persian_datetime.date.month().ordinal, 10);
//! assert_eq!(persian_datetime.date.day_of_month().0, 11);
//! assert_eq!(persian_datetime.time.hour.number(), 13);
//! assert_eq!(persian_datetime.time.minute.number(), 1);
//! assert_eq!(persian_datetime.time.second.number(), 0);
//! ```

use crate::any_calendar::AnyCalendarKind;
use crate::calendar_arithmetic::{ArithmeticDate, CalendarArithmetic};
use crate::helpers::{ceil_div, div_rem_euclid64, i64_to_i32, I32Result};
use crate::iso::Iso;
use crate::julian::Julian;
use crate::rata_die::RataDie;
use crate::{types, Calendar, CalendarError, Date, DateDuration, DateDurationUnit, DateTime};
use ::tinystr::tinystr;
use core::marker::PhantomData;

// Lisp code reference: https://github.com/EdReingold/calendar-code2/blob/main/calendar.l#L4720
// Book states that the Persian epoch is the date: 3/19/622 and since the Persian Calendar has no year 0, the best choice was to use the Julian function.
const FIXED_PERSIAN_EPOCH: RataDie = Julian::fixed_from_julian(ArithmeticDate {
    year: (622),
    month: (3),
    day: (19),
    marker: core::marker::PhantomData,
});
/// The Persian Calendar
///
/// The [Persian Calendar] is a solar calendar used officially by the countries of Iran and Afghanistan and many Persian-speaking regions.
/// It has 12 months and other similarities to the Gregorian Calendar
///
/// This type can be used with [`Date`] or [`DateTime`] to represent dates in this calendar.
///
/// [Persian Calendar]: https://en.wikipedia.org/wiki/Solar_Hijri_calendar
///
/// # Era codes
/// This calendar supports only one era code, which starts from the year of the Hijra, designated as "ah".
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq, PartialOrd, Ord)]
#[allow(clippy::exhaustive_structs)]
pub struct Persian;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]

/// The inner date type used for representing [`Date`]s of [`Persian`]. See [`Date`] and [`Persian`] for more details.
pub struct PersianDateInner(ArithmeticDate<Persian>);

impl CalendarArithmetic for Persian {
    fn month_days(year: i32, month: u8) -> u8 {
        match month {
            1 | 2 | 3 | 4 | 5 | 6 => 31,
            7 | 8 | 9 | 10 | 11 => 30,
            12 if Self::is_leap_year(year) => 30,
            12 => 29,
            _ => 0,
        }
    }

    fn months_for_every_year(_: i32) -> u8 {
        12
    }
    // Lisp code reference: https://github.com/EdReingold/calendar-code2/blob/main/calendar.l#L4789
    fn is_leap_year(p_year: i32) -> bool {
        let mut p_year = p_year as i64;
        if 0 < p_year {
            p_year -= 474;
        } else {
            p_year -= 473;
        };
        let d = div_rem_euclid64(p_year, 2820);
        let year = d.1 + 474;

        div_rem_euclid64((year + 38) * 31, 128).1 < 31
    }

    fn days_in_provided_year(year: i32) -> u32 {
        if Self::is_leap_year(year) {
            366
        } else {
            365
        }
    }

    fn last_month_day_in_year(year: i32) -> (u8, u8) {
        if Self::is_leap_year(year) {
            (12, 30)
        } else {
            (12, 29)
        }
    }
}

impl Calendar for Persian {
    type DateInner = PersianDateInner;
    fn date_from_codes(
        &self,
        era: types::Era,
        year: i32,
        month_code: types::MonthCode,
        day: u8,
    ) -> Result<Self::DateInner, CalendarError> {
        let year = if era.0 == tinystr!(16, "ah") {
            year
        } else {
            return Err(CalendarError::UnknownEra(era.0, self.debug_name()));
        };

        ArithmeticDate::new_from_solar(self, year, month_code, day).map(PersianDateInner)
    }

    fn date_from_iso(&self, iso: Date<Iso>) -> PersianDateInner {
        let fixed_iso = Iso::fixed_from_iso(*iso.inner());
        Self::arithmetic_persian_from_fixed(fixed_iso).inner
    }

    fn date_to_iso(&self, date: &Self::DateInner) -> Date<Iso> {
        let fixed_persian = Persian::fixed_from_arithmetic_persian(*date);
        Iso::iso_from_fixed(fixed_persian)
    }

    fn months_in_year(&self, date: &Self::DateInner) -> u8 {
        date.0.months_in_year()
    }

    fn days_in_year(&self, date: &Self::DateInner) -> u32 {
        date.0.days_in_year()
    }

    fn days_in_month(&self, date: &Self::DateInner) -> u8 {
        date.0.days_in_month()
    }

    fn day_of_week(&self, date: &Self::DateInner) -> types::IsoWeekday {
        Iso.day_of_week(self.date_to_iso(date).inner())
    }

    fn offset_date(&self, date: &mut Self::DateInner, offset: DateDuration<Self>) {
        date.0.offset_date(offset)
    }

    #[allow(clippy::field_reassign_with_default)]
    fn until(
        &self,
        date1: &Self::DateInner,
        date2: &Self::DateInner,
        _calendar2: &Self,
        _largest_unit: DateDurationUnit,
        _smallest_unit: DateDurationUnit,
    ) -> DateDuration<Self> {
        date1.0.until(date2.0, _largest_unit, _smallest_unit)
    }

    fn year(&self, date: &Self::DateInner) -> types::FormattableYear {
        Self::year_as_persian(date.0.year)
    }

    fn month(&self, date: &Self::DateInner) -> types::FormattableMonth {
        date.0.solar_month()
    }

    fn day_of_month(&self, date: &Self::DateInner) -> types::DayOfMonth {
        date.0.day_of_month()
    }

    fn day_of_year_info(&self, date: &Self::DateInner) -> types::DayOfYearInfo {
        let prev_year = date.0.year.saturating_sub(1);
        let next_year = date.0.year.saturating_add(1);
        types::DayOfYearInfo {
            day_of_year: date.0.day_of_year(),
            days_in_year: date.0.days_in_year(),
            prev_year: Persian::year_as_persian(prev_year),
            days_in_prev_year: Persian::days_in_provided_year(prev_year),
            next_year: Persian::year_as_persian(next_year),
        }
    }

    fn debug_name(&self) -> &'static str {
        "Persian"
    }
    // Missing any_calendar persian tests, the rest is completed
    fn any_calendar_kind(&self) -> Option<AnyCalendarKind> {
        Some(AnyCalendarKind::Persian)
    }
}

impl Persian {
    /// Constructs a new Persian Calendar
    pub fn new() -> Self {
        Self
    }

    // "Fixed" is a day count representation of calendars staring from Jan 1st of year 1 of the Georgian Calendar.
    // The fixed date algorithms are from
    // Dershowitz, Nachum, and Edward M. Reingold. _Calendrical calculations_. Cambridge University Press, 2008.
    //
    // Lisp code reference: https://github.com/EdReingold/calendar-code2/blob/main/calendar.l#L4803
    fn fixed_from_arithmetic_persian(p_date: PersianDateInner) -> RataDie {
        let p_year = i64::from(p_date.0.year);
        let month = i64::from(p_date.0.month);
        let day = i64::from(p_date.0.day);
        let y = if p_year > 0 {
            p_year - 474
        } else {
            p_year - 473
        };
        let x = div_rem_euclid64(y, 2820);
        let year = x.1 + 474;
        let z = div_rem_euclid64(31 * year - 5, 128);

        RataDie::new(
            FIXED_PERSIAN_EPOCH.to_i64_date() - 1
                + 1029983 * x.0
                + 365 * (year - 1)
                + z.0
                + if month <= 7 {
                    31 * (month - 1)
                } else {
                    30 * (month - 1) + 6
                }
                + day,
        )
    }
    // Lisp code reference: https://github.com/EdReingold/calendar-code2/blob/main/calendar.l#L4857
    fn arithmetic_persian_from_fixed(date: RataDie) -> Date<Persian> {
        let year = Self::arithmetic_persian_year_from_fixed(date);
        let year = match i64_to_i32(year) {
            I32Result::BelowMin(_) => {
                return Date::from_raw(PersianDateInner(ArithmeticDate::min_date()), Persian)
            }
            I32Result::AboveMax(_) => {
                return Date::from_raw(PersianDateInner(ArithmeticDate::max_date()), Persian)
            }
            I32Result::WithinRange(y) => y,
        };
        #[allow(clippy::unwrap_used)] // valid month,day
        let day_of_year = 1_i64 + (date - Self::fixed_from_persian_integers(year, 1, 1).unwrap());
        let month = if day_of_year <= 186 {
            ceil_div(day_of_year, 31) as u8
        } else {
            ceil_div(day_of_year - 6, 30) as u8
        };
        #[allow(clippy::unwrap_used)] // month and day
        let day = (date - Self::fixed_from_persian_integers(year, month, 1).unwrap() + 1) as u8;
        #[allow(clippy::unwrap_used)] // valid month and day
        Date::try_new_persian_date(year, month, day).unwrap()
    }

    // Lisp code reference: https://github.com/EdReingold/calendar-code2/blob/main/calendar.l#L4829
    fn arithmetic_persian_year_from_fixed(date: RataDie) -> i64 {
        #[allow(clippy::unwrap_used)] // valid year,month,day
        let d0 = date - Self::fixed_from_persian_integers(475, 1, 1).unwrap();
        let d = div_rem_euclid64(d0, 1029983);
        let n2820 = d.0;
        let d1 = d.1;
        let y2820 = if d1 == 1029982 {
            2820
        } else {
            div_rem_euclid64(128 * d1 + 46878, 46751).0
        };
        let year = 474 + n2820 * 2820 + y2820;
        if year > 0 {
            year
        } else {
            year - 1
        }
    }

    pub(crate) fn fixed_from_persian_integers(year: i32, month: u8, day: u8) -> Option<RataDie> {
        Date::try_new_persian_date(year, month, day)
            .ok()
            .map(|d| *d.inner())
            .map(Self::fixed_from_arithmetic_persian)
    }

    fn year_as_persian(year: i32) -> types::FormattableYear {
        types::FormattableYear {
            era: types::Era(tinystr!(16, "ah")),
            number: year,
            cyclic: None,
            related_iso: None,
        }
    }
}

impl Date<Persian> {
    /// Construct new Persian Date.
    ///
    /// Has no negative years, only era is the AH/AP.
    ///
    /// ```rust
    /// use icu::calendar::Date;
    ///
    /// let date_persian = Date::try_new_persian_date(1392, 4, 25)
    ///     .expect("Failed to initialize Persian Date instance.");
    ///
    /// assert_eq!(date_persian.year().number, 1392);
    /// assert_eq!(date_persian.month().ordinal, 4);
    /// assert_eq!(date_persian.day_of_month().0, 25);
    /// ```
    pub fn try_new_persian_date(
        year: i32,
        month: u8,
        day: u8,
    ) -> Result<Date<Persian>, CalendarError> {
        let inner = ArithmeticDate {
            year,
            month,
            day,
            marker: PhantomData,
        };

        let max_month = Persian::months_for_every_year(year);
        if month > max_month {
            return Err(CalendarError::Overflow {
                field: "month",
                max: max_month as usize,
            });
        }

        let max_day = Persian::month_days(year, month);
        if day > max_day {
            return Err(CalendarError::Overflow {
                field: "day",
                max: max_day as usize,
            });
        }
        Ok(Date::from_raw(PersianDateInner(inner), Persian))
    }
}

impl DateTime<Persian> {
    /// Construct a new Persian datetime from integers.
    ///
    /// ```rust
    /// use icu::calendar::DateTime;
    ///
    /// let datetime_persian = DateTime::try_new_persian_datetime(474, 10, 11, 13, 1, 0)
    ///     .expect("Failed to initialize Persian DateTime instance.");
    ///
    /// assert_eq!(datetime_persian.date.year().number, 474);
    /// assert_eq!(datetime_persian.date.month().ordinal, 10);
    /// assert_eq!(datetime_persian.date.day_of_month().0, 11);
    /// assert_eq!(datetime_persian.time.hour.number(), 13);
    /// assert_eq!(datetime_persian.time.minute.number(), 1);
    /// assert_eq!(datetime_persian.time.second.number(), 0);
    /// ```
    pub fn try_new_persian_datetime(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
    ) -> Result<DateTime<Persian>, CalendarError> {
        Ok(DateTime {
            date: Date::try_new_persian_date(year, month, day)?,
            time: types::Time::try_new(hour, minute, second, 0)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gregorian;
    #[derive(Debug)]
    struct DateCase {
        year: i32,
        month: u8,
        day: u8,
    }

    static TEST_FIXED_DATE: [i64; 33] = [
        -214193, -61387, 25469, 49217, 171307, 210155, 253427, 369740, 400085, 434355, 452605,
        470160, 473837, 507850, 524156, 544676, 567118, 569477, 601716, 613424, 626596, 645554,
        664224, 671401, 694799, 704424, 708842, 709409, 709580, 727274, 728714, 744313, 764652,
    ];

    static CASES: [DateCase; 33] = [
        DateCase {
            year: -1208,
            month: 5,
            day: 1,
        },
        DateCase {
            year: -790,
            month: 9,
            day: 14,
        },
        DateCase {
            year: -552,
            month: 7,
            day: 2,
        },
        DateCase {
            year: -487,
            month: 7,
            day: 9,
        },
        DateCase {
            year: -153,
            month: 10,
            day: 18,
        },
        DateCase {
            year: -46,
            month: 2,
            day: 30,
        },
        DateCase {
            year: 73,
            month: 8,
            day: 19,
        },
        DateCase {
            year: 392,
            month: 2,
            day: 5,
        },
        DateCase {
            year: 475,
            month: 3,
            day: 3,
        },
        DateCase {
            year: 569,
            month: 1,
            day: 3,
        },
        DateCase {
            year: 618,
            month: 12,
            day: 20,
        },
        DateCase {
            year: 667,
            month: 1,
            day: 14,
        },
        DateCase {
            year: 677,
            month: 2,
            day: 8,
        },
        DateCase {
            year: 770,
            month: 3,
            day: 22,
        },
        DateCase {
            year: 814,
            month: 11,
            day: 13,
        },
        DateCase {
            year: 871,
            month: 1,
            day: 21,
        },
        DateCase {
            year: 932,
            month: 6,
            day: 28,
        },
        DateCase {
            year: 938,
            month: 12,
            day: 14,
        },
        DateCase {
            year: 1027,
            month: 3,
            day: 21,
        },
        DateCase {
            year: 1059,
            month: 4,
            day: 10,
        },
        DateCase {
            year: 1095,
            month: 5,
            day: 2,
        },
        DateCase {
            year: 1147,
            month: 3,
            day: 30,
        },
        DateCase {
            year: 1198,
            month: 5,
            day: 10,
        },
        DateCase {
            year: 1218,
            month: 1,
            day: 7,
        },
        DateCase {
            year: 1282,
            month: 1,
            day: 29,
        },
        DateCase {
            year: 1308,
            month: 6,
            day: 3,
        },
        DateCase {
            year: 1320,
            month: 7,
            day: 7,
        },
        DateCase {
            year: 1322,
            month: 1,
            day: 29,
        },
        DateCase {
            year: 1322,
            month: 7,
            day: 14,
        },
        DateCase {
            year: 1370,
            month: 12,
            day: 27,
        },
        DateCase {
            year: 1374,
            month: 12,
            day: 6,
        },
        DateCase {
            year: 1417,
            month: 8,
            day: 19,
        },
        DateCase {
            year: 1473,
            month: 4,
            day: 28,
        },
    ];

    // Persian New Year occuring in March of Gregorian year (g_year) to fixed date
    fn nowruz(g_year: i32) -> RataDie {
        let iso_from_fixed: Date<Iso> =
            Iso::iso_from_fixed(RataDie::new(FIXED_PERSIAN_EPOCH.to_i64_date()));
        let greg_date_from_fixed: Date<Gregorian> = Date::new_from_iso(iso_from_fixed, Gregorian);
        let persian_year = g_year - greg_date_from_fixed.year().number + 1;
        let _year = if persian_year <= 0 {
            persian_year - 1
        } else {
            persian_year
        };
        #[allow(clippy::unwrap_used)] // valid day and month
        Persian::fixed_from_persian_integers(_year, 1, 1).unwrap()
    }

    fn days_in_provided_year_core(year: i32) -> u32 {
        #[allow(clippy::unwrap_used)] // valid month and day
        let fixed_year = Persian::fixed_from_persian_integers(year, 1, 1)
            .unwrap()
            .to_i64_date();
        #[allow(clippy::unwrap_used)] // valid month and day
        let next_fixed_year = Persian::fixed_from_persian_integers(year + 1, 1, 1)
            .unwrap()
            .to_i64_date();

        (next_fixed_year - fixed_year) as u32
    }

    #[test]
    fn test_persian_leap_year() {
        let mut leap_years: [i32; 33] = [0; 33];
        // These values were computed from the "Calendrical Calculations" reference code output
        let expected_values = [
            false, true, false, false, false, false, false, true, false, true, false, false, true,
            false, false, true, false, false, false, false, false, false, false, true, false,
            false, false, false, false, true, false, false, false,
        ];

        let mut leap_year_results: Vec<bool> = Vec::new();
        let canonical_leap_year_cycle_start = 474;
        let canonical_leap_year_cycle_end = 3293;
        for year in canonical_leap_year_cycle_start..=canonical_leap_year_cycle_end {
            let r = Persian::is_leap_year(year);
            if r {
                leap_year_results.push(r);
            }
        }
        // 683 is the amount of leap years in the 2820 Persian year cycle
        assert_eq!(leap_year_results.len(), 683);

        for (index, case) in CASES.iter().enumerate() {
            leap_years[index] = case.year;
        }
        for (year, bool) in leap_years.iter().zip(expected_values.iter()) {
            assert_eq!(Persian::is_leap_year(*year), *bool);
        }
    }

    #[test]
    fn days_in_provided_year_test() {
        for case in CASES.iter() {
            assert_eq!(
                days_in_provided_year_core(case.year),
                Persian::days_in_provided_year(case.year)
            );
        }
    }

    #[test]
    fn test_persian_year_from_fixed() {
        for (case, f_date) in CASES.iter().zip(TEST_FIXED_DATE.iter()) {
            let date = PersianDateInner(ArithmeticDate {
                year: (case.year),
                month: (case.month),
                day: (case.day),
                marker: (PhantomData),
            });
            assert_eq!(
                date.0.year as i64,
                Persian::arithmetic_persian_year_from_fixed(RataDie::new(*f_date)),
                "{case:?}"
            );
        }
    }
    #[test]
    fn test_fixed_from_persian() {
        for (case, f_date) in CASES.iter().zip(TEST_FIXED_DATE.iter()) {
            let date = PersianDateInner(ArithmeticDate {
                year: (case.year),
                month: (case.month),
                day: (case.day),
                marker: (PhantomData),
            });

            assert_eq!(
                Persian::fixed_from_arithmetic_persian(date).to_i64_date(),
                *f_date,
                "{case:?}"
            );
        }
    }
    #[test]
    fn test_persian_from_fixed() {
        for (case, f_date) in CASES.iter().zip(TEST_FIXED_DATE.iter()) {
            let date = Date::try_new_persian_date(case.year, case.month, case.day).unwrap();
            assert_eq!(
                Persian::arithmetic_persian_from_fixed(RataDie::new(*f_date)),
                date,
                "{case:?}"
            );
        }
    }
    #[test]
    fn test_persian_epoch() {
        let epoch = FIXED_PERSIAN_EPOCH.to_i64_date();
        // Iso year of Persian Epoch
        let epoch_year_from_fixed = Iso::iso_from_fixed(RataDie::new(epoch)).inner.0.year;
        // 622 is the correct ISO year for the Persian Epoch
        assert_eq!(epoch_year_from_fixed, 622);
    }

    #[test]
    fn test_nowruz() {
        let fixed_date = nowruz(622).to_i64_date();
        assert_eq!(fixed_date, FIXED_PERSIAN_EPOCH.to_i64_date());
        // These values are used as test data in appendix C of the "Calendrical Calculations" book
        let nowruz_test_year_start = 2000;
        let nowruz_test_year_end = 2103;

        for year in nowruz_test_year_start..=nowruz_test_year_end {
            let two_thousand_eight_to_fixed = nowruz(year).to_i64_date();
            let iso_date = Date::try_new_iso_date(year, 3, 21).unwrap();
            let persian_year =
                Persian::arithmetic_persian_year_from_fixed(Iso::fixed_from_iso(iso_date.inner));
            assert_eq!(
                Persian::arithmetic_persian_year_from_fixed(RataDie::new(
                    two_thousand_eight_to_fixed
                )),
                persian_year
            );
        }
    }

    #[test]
    fn test_day_of_year_info() {
        #[derive(Debug)]
        struct TestCase {
            input: i32,
            expected_prev: i32,
            expected_next: i32,
        }

        let test_cases = vec![
            TestCase {
                input: 0,
                expected_prev: -1,
                expected_next: 1,
            },
            TestCase {
                input: i32::MAX,
                expected_prev: i32::MAX - 1,
                expected_next: i32::MAX, // can't go above i32::MAX
            },
            TestCase {
                input: i32::MIN + 1,
                expected_prev: i32::MIN,
                expected_next: i32::MIN + 2,
            },
            TestCase {
                input: i32::MIN,
                expected_prev: i32::MIN, // can't go below i32::MIN
                expected_next: i32::MIN + 1,
            },
        ];

        for case in test_cases {
            let date = PersianDateInner(ArithmeticDate {
                year: (case.input),
                month: 1,
                day: 1,
                marker: PhantomData,
            });
            let info = Persian::day_of_year_info(&Persian, &date);

            assert_eq!(info.prev_year.number, case.expected_prev, "{:?}", case);
            assert_eq!(info.next_year.number, case.expected_next, "{:?}", case);
        }
    }
}