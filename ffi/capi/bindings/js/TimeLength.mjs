import wasm from "./diplomat-wasm.mjs"
import * as diplomatRuntime from "./diplomat-runtime.mjs"

export const TimeLength_js_to_rust = {
  "Full": 0,
  "Long": 1,
  "Medium": 2,
  "Short": 3,
};

export const TimeLength_rust_to_js = {
  [0]: "Full",
  [1]: "Long",
  [2]: "Medium",
  [3]: "Short",
};

export const TimeLength = {
  "Full": "Full",
  "Long": "Long",
  "Medium": "Medium",
  "Short": "Short",
};