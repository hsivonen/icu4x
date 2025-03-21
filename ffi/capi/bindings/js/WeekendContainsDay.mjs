// generated by diplomat-tool
import wasm from "./diplomat-wasm.mjs";
import * as diplomatRuntime from "./diplomat-runtime.mjs";


/** 
 * Documents which days of the week are considered to be a part of the weekend
 *
 * See the [Rust documentation for `weekend`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#method.weekend) for more information.
 */


export class WeekendContainsDay {
    
    #monday;
    
    get monday()  {
        return this.#monday;
    } 
    set monday(value) {
        this.#monday = value;
    }
    
    #tuesday;
    
    get tuesday()  {
        return this.#tuesday;
    } 
    set tuesday(value) {
        this.#tuesday = value;
    }
    
    #wednesday;
    
    get wednesday()  {
        return this.#wednesday;
    } 
    set wednesday(value) {
        this.#wednesday = value;
    }
    
    #thursday;
    
    get thursday()  {
        return this.#thursday;
    } 
    set thursday(value) {
        this.#thursday = value;
    }
    
    #friday;
    
    get friday()  {
        return this.#friday;
    } 
    set friday(value) {
        this.#friday = value;
    }
    
    #saturday;
    
    get saturday()  {
        return this.#saturday;
    } 
    set saturday(value) {
        this.#saturday = value;
    }
    
    #sunday;
    
    get sunday()  {
        return this.#sunday;
    } 
    set sunday(value) {
        this.#sunday = value;
    }
    
    /** Create `WeekendContainsDay` from an object that contains all of `WeekendContainsDay`s fields.
    * Optional fields do not need to be included in the provided object.
    */
    static fromFields(structObj) {
        return new WeekendContainsDay(structObj);
    }

    #internalConstructor(structObj) {
        if (typeof structObj !== "object") {
            throw new Error("WeekendContainsDay's constructor takes an object of WeekendContainsDay's fields.");
        }

        if ("monday" in structObj) {
            this.#monday = structObj.monday;
        } else {
            throw new Error("Missing required field monday.");
        }

        if ("tuesday" in structObj) {
            this.#tuesday = structObj.tuesday;
        } else {
            throw new Error("Missing required field tuesday.");
        }

        if ("wednesday" in structObj) {
            this.#wednesday = structObj.wednesday;
        } else {
            throw new Error("Missing required field wednesday.");
        }

        if ("thursday" in structObj) {
            this.#thursday = structObj.thursday;
        } else {
            throw new Error("Missing required field thursday.");
        }

        if ("friday" in structObj) {
            this.#friday = structObj.friday;
        } else {
            throw new Error("Missing required field friday.");
        }

        if ("saturday" in structObj) {
            this.#saturday = structObj.saturday;
        } else {
            throw new Error("Missing required field saturday.");
        }

        if ("sunday" in structObj) {
            this.#sunday = structObj.sunday;
        } else {
            throw new Error("Missing required field sunday.");
        }

        return this;
    }

    // Return this struct in FFI function friendly format.
    // Returns an array that can be expanded with spread syntax (...)
    
    _intoFFI(
        functionCleanupArena,
        appendArrayMap
    ) {
        return [this.#monday, this.#tuesday, this.#wednesday, this.#thursday, this.#friday, this.#saturday, this.#sunday]
    }

    static _fromSuppliedValue(internalConstructor, obj) {
        if (internalConstructor !== diplomatRuntime.internalConstructor) {
            throw new Error("_fromSuppliedValue cannot be called externally.");
        }

        if (obj instanceof WeekendContainsDay) {
            return obj;
        }

        return WeekendContainsDay.fromFields(obj);
    }

    _writeToArrayBuffer(
        arrayBuffer,
        offset,
        functionCleanupArena,
        appendArrayMap
    ) {
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 0, this.#monday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 1, this.#tuesday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 2, this.#wednesday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 3, this.#thursday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 4, this.#friday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 5, this.#saturday, Uint8Array);
        diplomatRuntime.writeToArrayBuffer(arrayBuffer, offset + 6, this.#sunday, Uint8Array);
    }

    // This struct contains borrowed fields, so this takes in a list of
    // "edges" corresponding to where each lifetime's data may have been borrowed from
    // and passes it down to individual fields containing the borrow.
    // This method does not attempt to handle any dependencies between lifetimes, the caller
    // should handle this when constructing edge arrays.
    static _fromFFI(internalConstructor, ptr) {
        if (internalConstructor !== diplomatRuntime.internalConstructor) {
            throw new Error("WeekendContainsDay._fromFFI is not meant to be called externally. Please use the default constructor.");
        }
        let structObj = {};
        const mondayDeref = (new Uint8Array(wasm.memory.buffer, ptr, 1))[0] === 1;
        structObj.monday = mondayDeref;
        const tuesdayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 1, 1))[0] === 1;
        structObj.tuesday = tuesdayDeref;
        const wednesdayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 2, 1))[0] === 1;
        structObj.wednesday = wednesdayDeref;
        const thursdayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 3, 1))[0] === 1;
        structObj.thursday = thursdayDeref;
        const fridayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 4, 1))[0] === 1;
        structObj.friday = fridayDeref;
        const saturdayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 5, 1))[0] === 1;
        structObj.saturday = saturdayDeref;
        const sundayDeref = (new Uint8Array(wasm.memory.buffer, ptr + 6, 1))[0] === 1;
        structObj.sunday = sundayDeref;

        return new WeekendContainsDay(structObj);
    }

    constructor(structObj) {
        return this.#internalConstructor(...arguments)
    }
}