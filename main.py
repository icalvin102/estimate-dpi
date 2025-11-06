#!/usr/bin/env python3
import argparse
import sys
from PIL import Image, TiffImagePlugin
import numpy as np

COMMON_THERMAL_DPIS = [152, 200, 203, 300, 305, 306, 400, 406, 600]

def get_scan_dpi(img, fallback):
    # Try multiple metadata places
    dpi = None
    # PIL stores as tuple for many formats (PPI)
    if "dpi" in img.info and isinstance(img.info["dpi"], (tuple, list)) and len(img.info["dpi"]) >= 1:
        dpi = float(img.info["dpi"][0])
    # TIFF can have resolution units and XResolution tags
    try:
        if hasattr(img, "tag_v2"):
            tags = img.tag_v2
            xres = tags.get(TiffImagePlugin.X_RESOLUTION)
            unit = tags.get(TiffImagePlugin.RESOLUTION_UNIT)  # 1: no unit, 2: inch, 3: cm
            if xres:
                x = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                if unit == 2 or unit is None:
                    dpi = float(x)
                elif unit == 3:  # per cm -> per inch
                    dpi = float(x) * 2.54
    except Exception:
        pass
    # PNG often uses "resolution" or "pHYs" (PIL maps to info.get("dpi") when possible)
    if dpi is None:
        dpi = fallback
    return float(dpi)

def hann(n):
    # Classic Hann window
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))

def quadratic_peak_interpolation(mags, k):
    # Parabolic interpolation around bin k (avoid edges)
    if k <= 0 or k >= len(mags) - 1:
        return k, mags[k]
    alpha, beta, gamma = mags[k-1], mags[k], mags[k+1]
    denom = (alpha - 2*beta + gamma)
    if denom == 0:
        return k, beta
    delta = 0.5 * (alpha - gamma) / denom  # shift in bins
    peak_mag = beta - 0.25 * (alpha - gamma) * delta
    return k + delta, peak_mag

def dominant_frequency_1d(signal, scan_dpi, dpi_min=100, dpi_max=1200):
    """
    signal: 1D float array (grayscale along one axis)
    Returns cycles-per-pixel frequency and strength.
    """
    n = len(signal)
    if n < 32:
        raise ValueError("Signal too short for FFT.")
    # Detrend & window
    s = signal.astype(np.float64)
    s -= np.mean(s)
    w = hann(n)
    s *= w
    # FFT
    spec = np.fft.rfft(s)
    mags = np.abs(spec)
    # Frequency bins in cycles per sample (pixel)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Convert target DPI bounds to cycles per pixel
    f_min = max(dpi_min / scan_dpi, 1.0 / n)  # at least > 0
    f_max = min(dpi_max / scan_dpi, 0.5)      # Nyquist
    # Mask range
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        raise ValueError("No valid frequency range after masking.")
    # Ignore DC and super low bins
    mags_masked = mags.copy()
    mags_masked[~mask] = 0.0
    # Find max (avoid 0 bin)
    k = int(np.argmax(mags_masked))
    # Refine with quadratic interpolation
    k_refined, peak_mag = quadratic_peak_interpolation(mags, k)
    f_peak = np.interp(k_refined, np.arange(len(freqs)), freqs)
    return f_peak, peak_mag

def sample_center_stripes(im_gray_np, along="x", thickness_px=16):
    H, W = im_gray_np.shape
    if along == "x":
        y0 = H//2 - thickness_px//2
        y1 = y0 + thickness_px
        y0 = max(0, y0); y1 = min(H, y1)
        strip = im_gray_np[y0:y1, :]
        return np.mean(strip, axis=0)  # length W
    else:
        x0 = W//2 - thickness_px//2
        x1 = x0 + thickness_px
        x0 = max(0, x0); x1 = min(W, x1)
        strip = im_gray_np[:, x0:x1]
        return np.mean(strip, axis=1)  # length H

def try_axes(im_arr, scan_dpi, dpi_min, dpi_max, thickness):
    # Try both directions; return the stronger (higher peak magnitude)
    row_sig = sample_center_stripes(im_arr, along="x", thickness_px=thickness)
    col_sig = sample_center_stripes(im_arr, along="y", thickness_px=thickness)
    fx, magx = dominant_frequency_1d(row_sig, scan_dpi, dpi_min, dpi_max)
    fy, magy = dominant_frequency_1d(col_sig, scan_dpi, dpi_min, dpi_max)
    if magx >= magy:
        return fx, "horizontal"
    else:
        return fy, "vertical"

def snap_common_dpi(raw_dpi):
    # Snap to nearest common thermal DPI if within a reasonable tolerance
    # Tolerance grows a bit with DPI due to noise
    candidates = np.array(COMMON_THERMAL_DPIS, dtype=float)
    idx = int(np.argmin(np.abs(candidates - raw_dpi)))
    nearest = candidates[idx]
    tol = max(6.0, 0.02 * nearest)  # e.g., ±6 dpi or ±2%
    if abs(nearest - raw_dpi) <= tol:
        return int(round(nearest))
    return int(round(raw_dpi))

def maybe_demote_harmonic(dpi_est):
    # If estimate looks like a clean 2x harmonic of a common DPI, halve it
    for base in COMMON_THERMAL_DPIS:
        if abs(dpi_est - 2*base) <= max(6.0, 0.02 * (2*base)):
            return int(round(dpi_est / 2))
    return dpi_est

def main():
    ap = argparse.ArgumentParser(description="Estimate thermal printer DPI from scanned image via FFT.")
    ap.add_argument("image", help="Path to the scanned image (ideally a small crop around the stripes).")
    ap.add_argument("--scan-dpi", type=float, default=None,
                    help="Scanner DPI (overrides image metadata). If not set, metadata is used; fallback 4800.")
    ap.add_argument("--thickness", type=int, default=24,
                    help="Rows/cols to average around the center to reduce noise (default: 24).")
    ap.add_argument("--dpi-min", type=int, default=100, help="Min expected DPI (default: 100).")
    ap.add_argument("--dpi-max", type=int, default=1200, help="Max expected DPI (default: 1200).")
    ap.add_argument("--axis", choices=["auto", "horizontal", "vertical"], default="auto",
                    help="Analyze along axis; 'horizontal' samples a central row across width. Default auto.")
    ap.add_argument("--no-snap", action="store_true",
                    help="Do not snap result to common thermal DPIs.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print diagnostics to stderr.")
    args = ap.parse_args()

    try:
        img = Image.open(args.image)
    except Exception as e:
        print(f"Error: cannot open image: {e}", file=sys.stderr)
        sys.exit(2)

    scan_dpi = get_scan_dpi(img, fallback=4800.0)
    if args.scan_dpi is not None:
        scan_dpi = float(args.scan_dpi)

    # Convert to grayscale float [0..1]
    im_gray = img.convert("L")
    im_arr = np.asarray(im_gray, dtype=np.float64) / 255.0

    try:
        if args.axis == "auto":
            f_cyc_per_px, direction = try_axes(im_arr, scan_dpi, args.dpi_min, args.dpi_max, args.thickness)
        elif args.axis == "horizontal":
            sig = sample_center_stripes(im_arr, along="x", thickness_px=args.thickness)
            f_cyc_per_px, _ = dominant_frequency_1d(sig, scan_dpi, args.dpi_min, args.dpi_max)
            direction = "horizontal"
        else:
            sig = sample_center_stripes(im_arr, along="y", thickness_px=args.thickness)
            f_cyc_per_px, _ = dominant_frequency_1d(sig, scan_dpi, args.dpi_min, args.dpi_max)
            direction = "vertical"
    except Exception as e:
        print(f"Error during frequency analysis: {e}", file=sys.stderr)
        sys.exit(3)

    raw_dpi = f_cyc_per_px * scan_dpi
    # Heuristic: if we likely hit a harmonic, demote it
    raw_dpi_harm_adjusted = maybe_demote_harmonic(raw_dpi)
    final_dpi = raw_dpi_harm_adjusted if args.no_snap else snap_common_dpi(raw_dpi_harm_adjusted)

    if args.verbose:
        print(f"# Diagnostics", file=sys.stderr)
        print(f"Scan DPI used: {scan_dpi:.2f}", file=sys.stderr)
        print(f"Direction chosen: {direction}", file=sys.stderr)
        print(f"Peak frequency: {f_cyc_per_px:.6f} cycles/pixel", file=sys.stderr)
        print(f"Raw DPI: {raw_dpi:.2f}", file=sys.stderr)
        if raw_dpi_harm_adjusted != int(round(raw_dpi)):
            print(f"Harmonic-adjusted DPI: {raw_dpi_harm_adjusted}", file=sys.stderr)
        if not args.no_snap:
            print(f"Snapped to common DPI: {final_dpi}", file=sys.stderr)

    # Print ONLY the most likely DPI to stdout (as requested)
    print(int(final_dpi))

if __name__ == "__main__":
    main()

