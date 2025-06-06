// generated by diplomat-tool
// dart format off

part of 'lib.g.dart';

/// See the [Rust documentation for `PluralCategory`](https://docs.rs/icu/2.0.0/icu/plurals/enum.PluralCategory.html) for more information.
enum PluralCategory {

  zero,

  one,

  two,

  few,

  many,

  other;

  /// Construct from a string in the format
  /// [specified in TR35](https://unicode.org/reports/tr35/tr35-numbers.html#Language_Plural_Rules)
  ///
  /// See the [Rust documentation for `get_for_cldr_string`](https://docs.rs/icu/2.0.0/icu/plurals/enum.PluralCategory.html#method.get_for_cldr_string) for more information.
  ///
  /// See the [Rust documentation for `get_for_cldr_bytes`](https://docs.rs/icu/2.0.0/icu/plurals/enum.PluralCategory.html#method.get_for_cldr_bytes) for more information.
  static PluralCategory? getForCldrString(String s) {
    final temp = _FinalizedArena();
    final result = _icu4x_PluralCategory_get_for_cldr_string_mv1(s._utf8AllocIn(temp.arena));
    if (!result.isOk) {
      return null;
    }
    return PluralCategory.values[result.union.ok];
  }

}

@_DiplomatFfiUse('icu4x_PluralCategory_get_for_cldr_string_mv1')
@ffi.Native<_ResultInt32Void Function(_SliceUtf8)>(isLeaf: true, symbol: 'icu4x_PluralCategory_get_for_cldr_string_mv1')
// ignore: non_constant_identifier_names
external _ResultInt32Void _icu4x_PluralCategory_get_for_cldr_string_mv1(_SliceUtf8 s);

// dart format on
