// generated by diplomat-tool
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";


/** See the [Rust documentation for `HangulSyllableType`](https://docs.rs/icu/latest/icu/properties/props/struct.HangulSyllableType.html) for more information.
*/


export class HangulSyllableType {
    

    static fromValue(value : HangulSyllableType | string) : HangulSyllableType; 

    get value() : string;

    get ffiValue() : number;

    static NotApplicable : HangulSyllableType;
    static LeadingJamo : HangulSyllableType;
    static VowelJamo : HangulSyllableType;
    static TrailingJamo : HangulSyllableType;
    static LeadingVowelSyllable : HangulSyllableType;
    static LeadingVowelTrailingSyllable : HangulSyllableType;

    static forChar(ch: codepoint): HangulSyllableType;

    toIntegerValue(): number;

    static fromIntegerValue(other: number): HangulSyllableType | null;

    constructor(value: HangulSyllableType | string );
}