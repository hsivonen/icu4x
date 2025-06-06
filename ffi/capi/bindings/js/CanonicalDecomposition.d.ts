// generated by diplomat-tool
import type { DataError } from "./DataError"
import type { DataProvider } from "./DataProvider"
import type { Decomposed } from "./Decomposed"
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";



/**
 * The raw (non-recursive) canonical decomposition operation.
 *
 * Callers should generally use DecomposingNormalizer unless they specifically need raw composition operations
 *
 * See the [Rust documentation for `CanonicalDecomposition`](https://docs.rs/icu/2.0.0/icu/normalizer/properties/struct.CanonicalDecomposition.html) for more information.
 */
export class CanonicalDecomposition {
    /** @internal */
    get ffiValue(): pointer;


    /**
     * Construct a new CanonicalDecomposition instance for NFC using a particular data source.
     *
     * See the [Rust documentation for `new`](https://docs.rs/icu/2.0.0/icu/normalizer/properties/struct.CanonicalDecomposition.html#method.new) for more information.
     */
    static createWithProvider(provider: DataProvider): CanonicalDecomposition;

    /**
     * Performs non-recursive canonical decomposition (including for Hangul).
     *
     * See the [Rust documentation for `decompose`](https://docs.rs/icu/2.0.0/icu/normalizer/properties/struct.CanonicalDecompositionBorrowed.html#method.decompose) for more information.
     */
    decompose(c: codepoint): Decomposed;

    /**
     * Construct a new CanonicalDecomposition instance for NFC using compiled data.
     *
     * See the [Rust documentation for `new`](https://docs.rs/icu/2.0.0/icu/normalizer/properties/struct.CanonicalDecomposition.html#method.new) for more information.
     */
    constructor();
}