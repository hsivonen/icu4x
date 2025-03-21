// generated by diplomat-tool
import type { pointer, codepoint } from "./diplomat-runtime.d.ts";


/** 
 * Documents which days of the week are considered to be a part of the weekend
 *
 * See the [Rust documentation for `weekend`](https://docs.rs/icu/latest/icu/calendar/week/struct.WeekCalculator.html#method.weekend) for more information.
 */
type WeekendContainsDay_obj = {
    monday: boolean;
    tuesday: boolean;
    wednesday: boolean;
    thursday: boolean;
    friday: boolean;
    saturday: boolean;
    sunday: boolean;
};



export class WeekendContainsDay {
    
    get monday() : boolean; 
    set monday(value: boolean); 
    
    get tuesday() : boolean; 
    set tuesday(value: boolean); 
    
    get wednesday() : boolean; 
    set wednesday(value: boolean); 
    
    get thursday() : boolean; 
    set thursday(value: boolean); 
    
    get friday() : boolean; 
    set friday(value: boolean); 
    
    get saturday() : boolean; 
    set saturday(value: boolean); 
    
    get sunday() : boolean; 
    set sunday(value: boolean); 
    
    /** Create `WeekendContainsDay` from an object that contains all of `WeekendContainsDay`s fields.
    * Optional fields do not need to be included in the provided object.
    */
    static fromFields(structObj : WeekendContainsDay_obj) : WeekendContainsDay;


    constructor(structObj : WeekendContainsDay_obj);
}