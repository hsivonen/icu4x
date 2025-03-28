// generated by diplomat-tool
import wasm from "./diplomat-wasm.mjs";
import * as diplomatRuntime from "./diplomat-runtime.mjs";


/** 
 * See the [Rust documentation for `BackwardSecondLevel`](https://docs.rs/icu/latest/icu/collator/options/enum.BackwardSecondLevel.html) for more information.
 */


export class CollatorBackwardSecondLevel {
    
    #value = undefined;

    static #values = new Map([
        ["Off", 0],
        ["On", 1]
    ]);

    static getAllEntries() {
        return CollatorBackwardSecondLevel.#values.entries();
    }
    
    #internalConstructor(value) {
        if (arguments.length > 1 && arguments[0] === diplomatRuntime.internalConstructor) {
            // We pass in two internalConstructor arguments to create *new*
            // instances of this type, otherwise the enums are treated as singletons.
            if (arguments[1] === diplomatRuntime.internalConstructor ) {
                this.#value = arguments[2];
                return this;
            }
            return CollatorBackwardSecondLevel.#objectValues[arguments[1]];
        }

        if (value instanceof CollatorBackwardSecondLevel) {
            return value;
        }

        let intVal = CollatorBackwardSecondLevel.#values.get(value);

        // Nullish check, checks for null or undefined
        if (intVal != null) {
            return CollatorBackwardSecondLevel.#objectValues[intVal];
        }

        throw TypeError(value + " is not a CollatorBackwardSecondLevel and does not correspond to any of its enumerator values.");
    }

    static fromValue(value) {
        return new CollatorBackwardSecondLevel(value);
    }

    get value() {
        return [...CollatorBackwardSecondLevel.#values.keys()][this.#value];
    }

    get ffiValue() {
        return this.#value;
    }
    static #objectValues = [
        new CollatorBackwardSecondLevel(diplomatRuntime.internalConstructor, diplomatRuntime.internalConstructor, 0),
        new CollatorBackwardSecondLevel(diplomatRuntime.internalConstructor, diplomatRuntime.internalConstructor, 1),
    ];

    static Off = CollatorBackwardSecondLevel.#objectValues[0];
    static On = CollatorBackwardSecondLevel.#objectValues[1];

    constructor(value) {
        return this.#internalConstructor(...arguments)
    }
}