import { Calendar } from "icu4x"
import { DataProvider } from "icu4x"
import { DateTime } from "icu4x"
import { IsoDateTime } from "icu4x"
import { Locale } from "icu4x"
import { Time } from "icu4x"
import { TimeFormatter } from "icu4x"
export function formatTime() {
    var terminusArgs = arguments;
    return (function (...args) { return args[0].formatTime(...args.slice(1)) }).apply(
        null,
        [
            TimeFormatter.createWithLength.apply(
                null,
                [
                    DataProvider.compiled.apply(
                        null,
                        [
                        ]
                    ),
                    Locale.fromString.apply(
                        null,
                        [
                            terminusArgs[0]
                        ]
                    ),
                    terminusArgs[1]
                ]
            ),
            Time.create.apply(
                null,
                [
                    terminusArgs[2],
                    terminusArgs[3],
                    terminusArgs[4],
                    terminusArgs[5]
                ]
            )
        ]
    );
}
export function formatDatetime() {
    var terminusArgs = arguments;
    return (function (...args) { return args[0].formatDatetime(...args.slice(1)) }).apply(
        null,
        [
            TimeFormatter.createWithLength.apply(
                null,
                [
                    DataProvider.compiled.apply(
                        null,
                        [
                        ]
                    ),
                    Locale.fromString.apply(
                        null,
                        [
                            terminusArgs[0]
                        ]
                    ),
                    terminusArgs[1]
                ]
            ),
            DateTime.fromIsoInCalendar.apply(
                null,
                [
                    terminusArgs[2],
                    terminusArgs[3],
                    terminusArgs[4],
                    terminusArgs[5],
                    terminusArgs[6],
                    terminusArgs[7],
                    terminusArgs[8],
                    Calendar.createForLocale.apply(
                        null,
                        [
                            DataProvider.compiled.apply(
                                null,
                                [
                                ]
                            ),
                            Locale.fromString.apply(
                                null,
                                [
                                    terminusArgs[9]
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    );
}
export function formatIsoDatetime() {
    var terminusArgs = arguments;
    return (function (...args) { return args[0].formatIsoDatetime(...args.slice(1)) }).apply(
        null,
        [
            TimeFormatter.createWithLength.apply(
                null,
                [
                    DataProvider.compiled.apply(
                        null,
                        [
                        ]
                    ),
                    Locale.fromString.apply(
                        null,
                        [
                            terminusArgs[0]
                        ]
                    ),
                    terminusArgs[1]
                ]
            ),
            IsoDateTime.create.apply(
                null,
                [
                    terminusArgs[2],
                    terminusArgs[3],
                    terminusArgs[4],
                    terminusArgs[5],
                    terminusArgs[6],
                    terminusArgs[7],
                    terminusArgs[8]
                ]
            )
        ]
    );
}