//! Rin Glow: an EffectCraft effect plug-in (plug-in API v1, see `docs/plugins.md`). Pixels
//! brighter than a threshold are blurred into a soft glow and added (or screened) back over the
//! layer, optionally tinted.
//!
//! * `lib.rs` (this file): the manifest and the WebAssembly ABI glue. You rarely need to touch it.
//! * `glow.rs`: the effect itself, plain safe Rust with tests (`cargo test`).
//!
//! Build: `cargo build --release --target wasm32-unknown-unknown`, then load
//! `target/wasm32-unknown-unknown/release/rin_glow.wasm` with Effect ▸ Load Effect Plug-in…
//! (or `effect.plugins.load {"path": …}`), or drop it in the settings folder's `Plug-ins`.

use std::cell::RefCell;

mod glow;

/// The manifest. Parameters are flattened in this order (10 values): threshold, radius,
/// intensity, color r g b a, colorize, composite, glow only.
pub const MANIFEST: &str = r#"{
  "api": 1,
  "id": "za.rin.glow",
  "name": "Rin Glow",
  "category": "Rin FX",
  "version": "1.0.0",
  "author": "Rin",
  "description": "Makes bright areas glow: thresholds the highlights, blurs them and adds them back.",
  "params": [
    {"id": "threshold", "name": "Threshold", "type": "slider", "default": 60, "min": 0, "max": 100, "decimals": 1},
    {"id": "radius", "name": "Radius", "type": "slider", "default": 30, "min": 0, "max": 1000, "sliderMax": 200, "decimals": 1},
    {"id": "intensity", "name": "Intensity", "type": "slider", "default": 1.5, "min": 0, "max": 20, "sliderMax": 5, "decimals": 2},
    {"id": "color", "name": "Glow Color", "type": "color", "default": [1.0, 1.0, 1.0, 1.0]},
    {"id": "colorize", "name": "Colorize", "type": "slider", "default": 0, "min": 0, "max": 100, "decimals": 1},
    {"id": "composite", "name": "Composite", "type": "popup", "options": ["Add", "Screen"], "default": 0},
    {"id": "glow_only", "name": "Glow Only", "type": "checkbox", "default": false}
  ]
}"#;

thread_local! {
    static BUF: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ec_api_version() -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn ec_manifest_ptr() -> i32 {
    MANIFEST.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn ec_manifest_len() -> i32 {
    MANIFEST.len() as i32
}

/// One reusable, 8-byte-aligned buffer for the parameters and pixels.
#[unsafe(no_mangle)]
pub extern "C" fn ec_alloc(bytes: i32) -> i32 {
    BUF.with(|b| {
        let mut b = b.borrow_mut();
        let words = (bytes.max(0) as usize).div_ceil(8);
        if b.len() < words {
            b.resize(words, 0);
        }
        b.as_mut_ptr() as i32
    })
}

/// Returns 0 on success; anything else leaves the frame unchanged.
///
/// # Safety
/// The host passes pointers into the buffer `ec_alloc` returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ec_render(pixels: i32, width: i32, height: i32, params: i32, nparams: i32, _time: f64, scale: f64) -> i32 {
    let (Ok(w), Ok(h), Ok(np)) = (usize::try_from(width), usize::try_from(height), usize::try_from(nparams)) else {
        return 1;
    };
    let Some(n) = w.checked_mul(h) else {
        return 1;
    };
    if pixels == 0 || params == 0 {
        return 1;
    }
    // The only unsafe code: viewing the host's buffer as typed slices.
    let p = unsafe { std::slice::from_raw_parts(params as *const f64, np) };
    let px = unsafe { std::slice::from_raw_parts_mut(pixels as *mut [f32; 4], n) };
    let Some(settings) = glow::Settings::from_flat(p, scale) else {
        return 2;
    };
    match glow::render(px, w, h, &settings) {
        Ok(()) => 0,
        Err(_) => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest is valid JSON and flattens to the values `glow::Settings::from_flat` reads.
    #[test]
    fn manifest_matches_the_flattened_params() {
        let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(m["api"], 1);
        let params = m["params"].as_array().unwrap();
        let ids: Vec<&str> = params.iter().map(|p| p["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["threshold", "radius", "intensity", "color", "colorize", "composite", "glow_only"]);
        let width = |t: &str| match t {
            "color" => 4,
            "point" => 2,
            _ => 1,
        };
        let n: usize = params.iter().map(|p| width(p["type"].as_str().unwrap())).sum();
        assert_eq!(n, glow::NPARAMS);
    }
}
