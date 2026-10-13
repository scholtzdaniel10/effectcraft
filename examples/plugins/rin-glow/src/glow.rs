//! The glow itself, in four steps:
//!
//! 1. **Extract**: keep the parts of each pixel brighter than the threshold (optionally tinted).
//! 2. **Blur**: spread them out (three box blurs ≈ a Gaussian blur, fast at any radius).
//! 3. **Scale**: multiply by the intensity.
//! 4. **Composite**: add or screen the glow over the original (or show the glow alone).
//!
//! Pixels are premultiplied RGBA `f32` (colour already multiplied by alpha).

/// How the glow combines with the layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Composite {
    Add,
    Screen,
}

/// The parameters, cleaned up and in buffer pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Luminance where the glow starts, 0–1.
    pub threshold: f32,
    /// How far the glow reaches, in buffer pixels.
    pub radius: f32,
    pub intensity: f32,
    pub color: [f32; 3],
    /// How much of the glow takes the glow colour, 0–1.
    pub colorize: f32,
    pub composite: Composite,
    pub glow_only: bool,
}

/// Number of flattened parameter values the manifest declares.
pub const NPARAMS: usize = 10;

impl Settings {
    /// Reads the flattened parameters (manifest order). `None` if there are too few.
    pub fn from_flat(p: &[f64], scale: f64) -> Option<Settings> {
        if p.len() < NPARAMS {
            return None;
        }
        // A value in range, or the default if the host sent something non-finite.
        let num = |i: usize, lo: f64, hi: f64, default: f64| match p.get(i) {
            Some(v) if v.is_finite() => v.clamp(lo, hi),
            _ => default,
        };
        let scale = if scale.is_finite() { scale.clamp(0.0, 16.0) } else { 1.0 };
        Some(Settings {
            threshold: (num(0, 0.0, 100.0, 60.0) / 100.0) as f32,
            radius: (num(1, 0.0, 1000.0, 30.0) * scale) as f32,
            intensity: num(2, 0.0, 20.0, 1.5) as f32,
            color: [num(3, 0.0, 1.0, 1.0) as f32, num(4, 0.0, 1.0, 1.0) as f32, num(5, 0.0, 1.0, 1.0) as f32],
            colorize: (num(7, 0.0, 100.0, 0.0) / 100.0) as f32,
            composite: if num(8, 0.0, 1.0e6, 0.0).round() == 1.0 { Composite::Screen } else { Composite::Add },
            glow_only: num(9, 0.0, 1.0, 0.0) > 0.5,
        })
    }
}

/// Applies the glow to `px` (`w × h` premultiplied RGBA pixels) in place.
pub fn render(px: &mut [[f32; 4]], w: usize, h: usize, s: &Settings) -> Result<(), &'static str> {
    if w.checked_mul(h) != Some(px.len()) {
        return Err("the buffer isn't width × height pixels");
    }
    if px.is_empty() {
        return Ok(());
    }
    // 1. Extract the highlights.
    let mut glow: Vec<[f32; 4]> = px.iter().map(|p| extract(*p, s)).collect();
    // 2. Blur them. Radius ≈ 3 standard deviations, about where a Gaussian fades out.
    blur(&mut glow, w, h, s.radius as f64 / 3.0);
    // 3 + 4. Scale and composite.
    for (p, g) in px.iter_mut().zip(&glow) {
        *p = composite(*p, g.map(|c| c * s.intensity), s);
    }
    Ok(())
}

/// Rec. 709 luminance of a colour (premultiplied in, premultiplied out).
fn luma(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Step 1: the part of a pixel that glows. Below the threshold nothing; from the threshold up to
/// white it fades in linearly; then optionally tinted towards the glow colour.
fn extract(c: [f32; 4], s: &Settings) -> [f32; 4] {
    let a = c[3];
    // Transparent or broken (NaN) pixels never glow.
    if a.is_nan() || a <= 0.0 {
        return [0.0; 4];
    }
    let y = luma(c) / a; // un-premultiplied luminance: how bright the colour itself is
    let ramp = (1.0 - s.threshold).max(1e-3);
    let weight = ((y - s.threshold) / ramp).clamp(0.0, 1.0);
    if weight.is_nan() || weight <= 0.0 {
        return [0.0; 4];
    }
    // Scaling all four channels keeps the pixel premultiplied.
    let mut g = c.map(|v| v * weight);
    if s.colorize > 0.0 {
        // The glow colour at the same brightness, mixed in by `colorize`.
        let y = luma(g);
        for (ch, tint) in g.iter_mut().zip(s.color) {
            *ch += (tint * y - *ch) * s.colorize;
        }
    }
    g
}

/// Step 2: a Gaussian blur of `sigma` buffer pixels.
///
/// A glow is soft, so big ones are blurred on a smaller copy of the image and enlarged again
/// smoothly: `k` times smaller means `k²` times less work, and the difference doesn't show.
fn blur(buf: &mut [[f32; 4]], w: usize, h: usize, sigma: f64) {
    let k = downsample_factor(sigma);
    if k <= 1 || w == 0 || h == 0 {
        blur_full(buf, w, h, sigma);
        return;
    }
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));

    // Shrink: each small pixel is the average of its k × k block (outside counts as transparent).
    let mut small = vec![[0.0f32; 4]; sw * sh];
    let share = 1.0 / (k * k) as f32;
    for (y, row) in buf.chunks(w).enumerate() {
        let small_row = (y / k) * sw;
        for (x, p) in row.iter().enumerate() {
            if let Some(s) = small.get_mut(small_row + x / k) {
                for (s, c) in s.iter_mut().zip(p) {
                    *s += c * share;
                }
            }
        }
    }

    // Blur the small copy. Shrinking and enlarging soften a little too (about 0.25 small
    // pixels² of variance), so blur that much less.
    let small_sigma = sigma / k as f64;
    blur_full(&mut small, sw, sh, (small_sigma * small_sigma - 0.25).max(0.0).sqrt());

    // Enlarge with bilinear interpolation between the four nearest small pixels.
    let at = |x: usize, y: usize| small.get(y * sw + x).copied().unwrap_or([0.0; 4]);
    // Where full-size pixel `i` falls between small pixels: (lower, upper, weight of upper).
    let place = |i: usize, n: usize| {
        let pos = ((i as f64 + 0.5) / k as f64 - 0.5).clamp(0.0, n.saturating_sub(1) as f64);
        let lo = pos.floor() as usize;
        (lo, (lo + 1).min(n.saturating_sub(1)), (pos - lo as f64) as f32)
    };
    for (y, row) in buf.chunks_mut(w).enumerate() {
        let (y0, y1, fy) = place(y, sh);
        for (x, p) in row.iter_mut().enumerate() {
            let (x0, x1, fx) = place(x, sw);
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for (i, out) in p.iter_mut().enumerate() {
                let top = a[i] + (b[i] - a[i]) * fx;
                let bottom = c[i] + (d[i] - c[i]) * fx;
                *out = top + (bottom - top) * fy;
            }
        }
    }
}

/// How many times smaller to make the image before blurring (1 = full resolution): a third of
/// sigma keeps the blur at least 3 small pixels wide, at most 16×.
fn downsample_factor(sigma: f64) -> usize {
    if !sigma.is_finite() {
        return 1;
    }
    ((sigma / 3.0).floor().max(1.0) as usize).clamp(1, 16)
}

/// Three box blurs in a row look almost exactly like a Gaussian blur, and each costs the same at
/// any radius (a running sum).
fn blur_full(buf: &mut [[f32; 4]], w: usize, h: usize, sigma: f64) {
    if w == 0 || h == 0 || sigma.is_nan() || sigma <= 0.0 {
        return;
    }
    let radii = box_radii(sigma);
    let mut line: Vec<[f32; 4]> = Vec::with_capacity(w.max(h));
    let mut scratch: Vec<[f32; 4]> = Vec::with_capacity(w.max(h));
    // Horizontal: each row.
    for row in buf.chunks_mut(w) {
        blur_line(row, &mut scratch, &radii);
    }
    // Vertical: each column, copied out and back.
    for x in 0..w {
        line.clear();
        line.extend(buf.iter().skip(x).step_by(w));
        blur_line(&mut line, &mut scratch, &radii);
        for (p, v) in buf.iter_mut().skip(x).step_by(w).zip(&line) {
            *p = *v;
        }
    }
}

/// The half-widths of three box blurs that together match a Gaussian of `sigma` (the standard
/// "boxes for Gauss" recipe: two widths, `m` boxes of the smaller one).
fn box_radii(sigma: f64) -> [usize; 3] {
    const N: f64 = 3.0;
    let var12 = 12.0 * sigma * sigma;
    let ideal = (var12 / N + 1.0).sqrt();
    let mut wl = ideal.floor().max(1.0);
    if wl % 2.0 == 0.0 {
        wl -= 1.0;
    }
    let wu = wl + 2.0;
    let m = ((var12 - N * wl * wl - 4.0 * N * wl - 3.0 * N) / (-4.0 * wl - 4.0)).round();
    let half = |width: f64| ((width - 1.0) / 2.0).max(0.0) as usize;
    [0.0, 1.0, 2.0].map(|i| if i < m { half(wl) } else { half(wu) })
}

/// Box-blurs one row or column in place, once per radius.
fn blur_line(line: &mut [[f32; 4]], scratch: &mut Vec<[f32; 4]>, radii: &[usize; 3]) {
    for &r in radii {
        scratch.clear();
        scratch.extend_from_slice(line);
        box_pass(scratch, line, r);
    }
}

/// One box blur of half-width `r`: each output is the average of the `2r + 1` inputs around it.
/// Outside the buffer counts as transparent, so light near the edge fades out instead of piling
/// up.
fn box_pass(src: &[[f32; 4]], dst: &mut [[f32; 4]], r: usize) {
    if r == 0 {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s;
        }
        return;
    }
    let norm = 1.0 / (2.0 * r as f64 + 1.0);
    // Running sum of the window (f64 so long rows don't drift), and how many lit pixels are in
    // it: when none are, the output is exactly 0 rather than rounding dust.
    let mut sum = [0.0f64; 4];
    let mut lit = 0usize;
    let lit_px = |p: &[f32; 4]| p.iter().any(|c| *c != 0.0);
    for p in src.iter().take(r) {
        add(&mut sum, p, 1.0);
        lit += usize::from(lit_px(p));
    }
    for (i, d) in dst.iter_mut().enumerate() {
        if let Some(p) = i.checked_add(r).and_then(|j| src.get(j)) {
            add(&mut sum, p, 1.0);
            lit += usize::from(lit_px(p));
        }
        if lit == 0 {
            sum = [0.0; 4];
            *d = [0.0; 4];
        } else {
            *d = sum.map(|c| (c * norm).max(0.0) as f32);
        }
        if let Some(p) = i.checked_sub(r).and_then(|j| src.get(j)) {
            add(&mut sum, p, -1.0);
            lit = lit.saturating_sub(usize::from(lit_px(p)));
        }
    }
}

fn add(sum: &mut [f64; 4], p: &[f32; 4], sign: f64) {
    for (s, c) in sum.iter_mut().zip(p) {
        *s += sign * *c as f64;
    }
}

/// Step 4: combine the (already intensity-scaled) glow `g` with the original pixel `o`.
fn composite(o: [f32; 4], g: [f32; 4], s: &Settings) -> [f32; 4] {
    let ga = g[3].clamp(0.0, 1.0);
    if s.glow_only {
        return [g[0], g[1], g[2], ga];
    }
    match s.composite {
        // Light adds up (can go brighter than white, like real light); coverage combines like
        // two layers over each other.
        Composite::Add => [o[0] + g[0], o[1] + g[1], o[2] + g[2], o[3] + ga * (1.0 - o[3]).max(0.0)],
        // Screen: 1 − (1 − a)(1 − b). Never brighter than white for normal pixels.
        Composite::Screen => {
            let screen = |a: f32, b: f32| {
                let b = b.clamp(0.0, 1.0);
                a + b - a * b
            };
            [screen(o[0], g[0]), screen(o[1], g[1]), screen(o[2], g[2]), screen(o[3], ga)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest's defaults, flattened.
    fn defaults() -> Vec<f64> {
        vec![60.0, 30.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    }

    #[test]
    fn reads_defaults_as_fractions_and_buffer_pixels() {
        let s = Settings::from_flat(&defaults(), 1.0).unwrap();
        assert_eq!(s.threshold, 0.6);
        assert_eq!(s.radius, 30.0);
        assert_eq!(s.intensity, 1.5);
        assert_eq!(s.color, [1.0, 1.0, 1.0]);
        assert_eq!(s.colorize, 0.0);
        assert_eq!(s.composite, Composite::Add);
        assert!(!s.glow_only);
    }

    #[test]
    fn radius_follows_preview_scale() {
        let s = Settings::from_flat(&defaults(), 0.5).unwrap();
        assert_eq!(s.radius, 15.0);
    }

    #[test]
    fn reads_screen_and_glow_only() {
        let mut p = defaults();
        p[8] = 1.0;
        p[9] = 1.0;
        let s = Settings::from_flat(&p, 1.0).unwrap();
        assert_eq!(s.composite, Composite::Screen);
        assert!(s.glow_only);
    }

    #[test]
    fn rejects_too_few_params() {
        assert!(Settings::from_flat(&defaults()[..9], 1.0).is_none());
    }

    #[test]
    fn bad_numbers_fall_back_to_safe_values() {
        let p = vec![f64::NAN, -5.0, f64::INFINITY, 2.0, -1.0, f64::NAN, 1.0, 500.0, 7.0, 0.0];
        let s = Settings::from_flat(&p, f64::NAN).unwrap();
        assert_eq!(s.threshold, 0.6); // NaN → default
        assert_eq!(s.radius, 0.0); // negative → clamped; NaN scale → 1
        assert_eq!(s.intensity, 1.5); // infinite → default
        assert_eq!(s.color, [1.0, 0.0, 1.0]); // clamped to 0–1, NaN → 1
        assert_eq!(s.colorize, 1.0); // 500 % → 100 %
        assert_eq!(s.composite, Composite::Add); // unknown popup index → Add
    }

    // ------------------------------------------------------------------ render

    const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const CLEAR: [f32; 4] = [0.0; 4];

    /// Default settings with a few changes.
    fn settings(change: impl FnOnce(&mut Settings)) -> Settings {
        let mut s = Settings::from_flat(&defaults(), 1.0).unwrap();
        change(&mut s);
        s
    }

    /// A `size × size` image of `fill` with one `dot` in the middle.
    fn dot_image(size: usize, fill: [f32; 4], dot: [f32; 4]) -> Vec<[f32; 4]> {
        let mut px = vec![fill; size * size];
        px[(size / 2) * size + size / 2] = dot;
        px
    }

    /// The pixel `dx` to the right of the middle.
    fn right_of_middle(px: &[[f32; 4]], size: usize, dx: usize) -> [f32; 4] {
        px[(size / 2) * size + size / 2 + dx]
    }

    fn glowed(size: usize, fill: [f32; 4], dot: [f32; 4], s: &Settings) -> Vec<[f32; 4]> {
        let mut px = dot_image(size, fill, dot);
        render(&mut px, size, size, s).unwrap();
        px
    }

    #[test]
    fn pixels_below_the_threshold_stay_unchanged() {
        let grey = [0.3, 0.3, 0.3, 1.0];
        let mut px = vec![grey; 64];
        render(&mut px, 8, 8, &settings(|_| {})).unwrap();
        assert!(px.iter().all(|p| *p == grey));
    }

    #[test]
    fn a_bright_pixel_glows_onto_its_neighbours() {
        let s = settings(|s| s.radius = 6.0);
        let px = glowed(41, BLACK, WHITE, &s);
        assert!(right_of_middle(&px, 41, 3)[0] > 0.0);
    }

    #[test]
    fn the_glow_fades_with_distance_and_stops_past_the_radius() {
        let s = settings(|s| s.radius = 6.0);
        let px = glowed(41, BLACK, WHITE, &s);
        let near = right_of_middle(&px, 41, 1)[0];
        let mid = right_of_middle(&px, 41, 4)[0];
        let far = right_of_middle(&px, 41, 12)[0];
        assert!(near > mid && mid > 0.0, "near {near} mid {mid}");
        assert_eq!(far, 0.0);
    }

    #[test]
    fn a_bigger_radius_reaches_further() {
        let small = glowed(81, BLACK, WHITE, &settings(|s| s.radius = 6.0));
        let big = glowed(81, BLACK, WHITE, &settings(|s| s.radius = 24.0));
        assert_eq!(right_of_middle(&small, 81, 15)[0], 0.0);
        assert!(right_of_middle(&big, 81, 15)[0] > 0.0);
    }

    #[test]
    fn zero_radius_brightens_only_the_bright_pixel() {
        let px = glowed(9, BLACK, WHITE, &settings(|s| s.radius = 0.0));
        assert!(right_of_middle(&px, 9, 0)[0] > 1.0);
        assert_eq!(right_of_middle(&px, 9, 1), BLACK);
    }

    #[test]
    fn zero_intensity_leaves_the_layer_unchanged() {
        let px = glowed(21, BLACK, WHITE, &settings(|s| s.intensity = 0.0));
        assert_eq!(px, dot_image(21, BLACK, WHITE));
    }

    #[test]
    fn the_blur_keeps_the_glows_total_light() {
        let s = settings(|s| {
            s.threshold = 0.0;
            s.intensity = 1.0;
            s.radius = 9.0;
            s.glow_only = true;
        });
        let px = glowed(61, CLEAR, WHITE, &s);
        assert!(right_of_middle(&px, 61, 3)[0] > 0.0, "the light must spread");
        let total: f32 = px.iter().map(|p| p[0]).sum();
        assert!((total - 1.0).abs() < 1e-3, "total {total}");
    }

    #[test]
    fn the_glow_spreads_into_transparent_areas_as_valid_premultiplied_colour() {
        let px = glowed(41, CLEAR, WHITE, &settings(|s| s.radius = 6.0));
        let p = right_of_middle(&px, 41, 2);
        assert!(p[3] > 0.0, "alpha {}", p[3]);
        assert!(px.iter().all(|p| p[3] <= 1.0));
        // Where the result is still see-through, colour never exceeds coverage. (Fully opaque
        // pixels may go brighter than white: that's HDR, as in a 32 bpc project.)
        assert!(px.iter().filter(|p| p[3] < 1.0).all(|p| p[0] <= p[3] + 1e-6));
    }

    #[test]
    fn colorize_tints_the_glow() {
        let s = settings(|s| {
            s.color = [1.0, 0.0, 0.0];
            s.colorize = 1.0;
            s.radius = 6.0;
        });
        let p = right_of_middle(&glowed(41, BLACK, WHITE, &s), 41, 3);
        assert!(p[0] > 0.0);
        assert!(p[1].abs() < 1e-6 && p[2].abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn glow_only_hides_the_layer() {
        let grey = [0.3, 0.3, 0.3, 1.0];
        let s = settings(|s| {
            s.radius = 4.0;
            s.glow_only = true;
        });
        let px = glowed(41, grey, WHITE, &s);
        assert_eq!(px[0], CLEAR);
        assert!(right_of_middle(&px, 41, 1)[0] > 0.0);
    }

    #[test]
    fn screen_never_goes_above_one() {
        let s = settings(|s| {
            s.composite = Composite::Screen;
            s.intensity = 20.0;
            s.radius = 4.0;
        });
        let px = glowed(21, [0.5, 0.5, 0.5, 1.0], WHITE, &s);
        assert!(px.iter().flatten().all(|c| *c <= 1.0));
        assert!(right_of_middle(&px, 21, 1)[0] > 0.5);
    }

    #[test]
    fn a_buffer_of_the_wrong_size_is_refused() {
        let mut px = vec![WHITE; 10];
        assert!(render(&mut px, 4, 4, &settings(|_| {})).is_err());
    }

    #[test]
    fn tiny_images_and_huge_radii_dont_crash() {
        let s = settings(|s| s.radius = 16_000.0);
        let mut empty: Vec<[f32; 4]> = vec![];
        render(&mut empty, 0, 0, &s).unwrap();
        let mut one = vec![WHITE];
        render(&mut one, 1, 1, &s).unwrap();
        let px = glowed(5, BLACK, WHITE, &s);
        assert!(px.iter().flatten().all(|c| c.is_finite()));
    }

    // ------------------------------------------------------- reduced-resolution blur

    /// A transparent frame with a few things that glow, well away from the edges.
    fn scene(w: usize, h: usize) -> Vec<[f32; 4]> {
        let mut px = vec![CLEAR; w * h];
        for y in 40..46 {
            for x in 40..46 {
                px[y * w + x] = WHITE;
            }
        }
        px[60 * w + 100] = [2.0, 2.0, 2.0, 1.0]; // an HDR highlight
        px[70 * w + 70] = [1.0, 0.0, 0.0, 1.0];
        px
    }

    #[test]
    fn small_glows_blur_at_full_resolution() {
        assert_eq!(downsample_factor(2.0), 1);
        assert_eq!(downsample_factor(5.9), 1);
    }

    #[test]
    fn big_glows_blur_at_reduced_resolution_up_to_16x() {
        assert_eq!(downsample_factor(12.0), 4);
        assert_eq!(downsample_factor(66.0), 16);
        assert_eq!(downsample_factor(1.0e9), 16);
    }

    #[test]
    fn reduced_resolution_blur_matches_full_resolution() {
        let (w, h) = (150, 110);
        for sigma in [12.0, 20.0] {
            let mut fast = scene(w, h);
            blur(&mut fast, w, h, sigma);
            let mut exact = scene(w, h);
            blur_full(&mut exact, w, h, sigma);
            let peak = exact.iter().flatten().fold(0.0f32, |m, c| m.max(*c));
            let worst = fast.iter().zip(&exact).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs())).fold(0.0f32, f32::max);
            assert!(worst < 0.1 * peak, "sigma {sigma}: worst {worst} vs peak {peak}");
            for c in 0..4 {
                let (f, e): (f32, f32) = (fast.iter().map(|p| p[c]).sum(), exact.iter().map(|p| p[c]).sum());
                assert!((f - e).abs() < 0.01 * e, "sigma {sigma} channel {c}: total {f} vs {e}");
            }
        }
    }

    #[test]
    fn odd_sized_images_blur_at_reduced_resolution_without_crashing() {
        for (w, h) in [(1, 1), (1, 37), (37, 1), (17, 5)] {
            let mut px = vec![WHITE; w * h];
            blur(&mut px, w, h, 30.0);
            assert!(px.iter().flatten().all(|c| c.is_finite() && *c >= 0.0));
        }
    }
}
