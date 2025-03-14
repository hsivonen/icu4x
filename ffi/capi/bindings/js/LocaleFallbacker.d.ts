// generated by diplomat-tool
import type { DataError } from "./DataError"
import type { DataProvider } from "./DataProvider"
import type { LocaleFallbackConfig } from "./LocaleFallbackConfig"
import type { LocaleFallbackConfig_obj } from "./LocaleFallbackConfig"
import type { LocaleFallbackerWithConfig } from "./LocaleFallbackerWithConfig"
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";


/** An object that runs the ICU4X locale fallback algorithm.
*
*See the [Rust documentation for `LocaleFallbacker`](https://docs.rs/icu/latest/icu/locale/fallback/struct.LocaleFallbacker.html) for more information.
*/


export class LocaleFallbacker {
    
    get ffiValue(): pointer;

    static createWithProvider(provider: DataProvider): LocaleFallbacker;

    static withoutData(): LocaleFallbacker;

    forConfig(config: LocaleFallbackConfig_obj): LocaleFallbackerWithConfig;

    constructor();
}