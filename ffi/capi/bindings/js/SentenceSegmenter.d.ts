// generated by diplomat-tool
import type { DataError } from "./DataError"
import type { DataProvider } from "./DataProvider"
import type { Locale } from "./Locale"
import type { SentenceBreakIteratorUtf16 } from "./SentenceBreakIteratorUtf16"
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";


/** An ICU4X sentence-break segmenter, capable of finding sentence breakpoints in strings.
*
*See the [Rust documentation for `SentenceSegmenter`](https://docs.rs/icu/latest/icu/segmenter/struct.SentenceSegmenter.html) for more information.
*/


export class SentenceSegmenter {
    
    get ffiValue(): pointer;

    static createWithContentLocale(locale: Locale): SentenceSegmenter;

    static createWithContentLocaleAndProvider(provider: DataProvider, locale: Locale): SentenceSegmenter;

    segment(input: string): SentenceBreakIteratorUtf16;

    constructor();
}