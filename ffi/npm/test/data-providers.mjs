// This file is part of ICU4X. For terms of use, please see the file
// called LICENSE at the top level of the ICU4X source tree
// (online at: https://github.com/unicode-org/icu4x/blob/main/LICENSE ).

import test from 'ava';

import { Decimal, Locale, DecimalFormatter, DecimalGroupingStrategy } from 'icu4x';

test('use createCompiled to format a simple decimal', async t => {
  const locale = Locale.fromString('bn');

  const format = DecimalFormatter.createWithGroupingStrategy(locale, DecimalGroupingStrategy.Auto);

  const decimal = Decimal.fromNumber(1234);
  decimal.multiplyPow10(-2);

  t.is(format.format(decimal), '১২.৩৪');
});
