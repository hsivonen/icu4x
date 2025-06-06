// generated by diplomat-tool
import type { DataError } from "./DataError"
import type { DataProvider } from "./DataProvider"
import type { TimeZone } from "./TimeZone"
import type { TimeZoneIterator } from "./TimeZoneIterator"
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";



/**
 * A mapper between IANA time zone identifiers and BCP-47 time zone identifiers.
 *
 * This mapper supports two-way mapping, but it is optimized for the case of IANA to BCP-47.
 * It also supports normalizing and canonicalizing the IANA strings.
 *
 * See the [Rust documentation for `IanaParser`](https://docs.rs/icu/2.0.0/icu/time/zone/iana/struct.IanaParser.html) for more information.
 */
export class IanaParser {
    /** @internal */
    get ffiValue(): pointer;


    /**
     * Create a new {@link IanaParser} using a particular data source
     *
     * See the [Rust documentation for `new`](https://docs.rs/icu/2.0.0/icu/time/zone/iana/struct.IanaParser.html#method.new) for more information.
     */
    static createWithProvider(provider: DataProvider): IanaParser;

    /**
     * See the [Rust documentation for `parse`](https://docs.rs/icu/2.0.0/icu/time/zone/iana/struct.IanaParserBorrowed.html#method.parse) for more information.
     */
    parse(value: string): TimeZone;

    /**
     * See the [Rust documentation for `iter`](https://docs.rs/icu/2.0.0/icu/time/zone/iana/struct.IanaParserBorrowed.html#method.iter) for more information.
     */
    iter(): TimeZoneIterator;

    /**
     * Create a new {@link IanaParser} using compiled data
     *
     * See the [Rust documentation for `new`](https://docs.rs/icu/2.0.0/icu/time/zone/iana/struct.IanaParser.html#method.new) for more information.
     */
    constructor();
}