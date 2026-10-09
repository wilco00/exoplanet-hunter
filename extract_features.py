"""Milestone 4b: compute vetting features for one KOI, using NASA's catalog ephemeris.

Each feature targets a way a false positive gives itself away:

  depth_ppm, depth_snr   how deep and how significant the dip is
  odd_even_sigma         an eclipsing binary seen at half its true period alternates
  odd_even_rel           deep/shallow eclipses; planets don't. sigma = statistical
                         significance, rel = |odd - even| / depth (size of the difference)
  secondary_snr          a dip at phase 0.5 means the "planet" glows -> likely a star.
  secondary_ratio        snr = significance, ratio = secondary depth / primary depth
  shape_ratio            edge depth / center depth: ~1 for a flat-bottomed (U) planet
                         transit, ~0.4 for a V-shaped grazing eclipse
  harmonic_dll           how much better a sine wave fits than a transit: positive means
                         it's probably stellar variability, not a transit
  max_single_frac        share of the signal coming from the single strongest event:
                         near 1 means one glitch is faking a periodic signal
  duration_frac          duration / period: very large values suggest stellar eclipses
  n_transits, noise_ppm  how much data backs the signal, and the out-of-transit scatter

Why both sigma and relative versions: at very high SNR, even a 1% odd/even difference
is "significant", so a bright, real planet can trip a sigma-only test.

If the star hosts other KOIs, their transits are left out of the trend fit and removed
before computing this KOI's features, so they can't contaminate it.

Usage:
    python extract_features.py --koi K00072.01     # Kepler-10b (confirmed planet)
    python extract_features.py --koi K01133.01     # an eclipsing-binary false positive
"""
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.timeseries import BoxLeastSquares  # noqa: E402

from build_koi_list import CATALOG_PATH  # noqa: E402
from clean_lightcurve import DATA_COLOR, INK, SURFACE, TREND_COLOR, clean, style  # noqa: E402
from fetch_lightcurve import OUT_DIR, fetch  # noqa: E402
from search_transits import POINT_COLOR, binned_median, fold_hours  # noqa: E402

WINDOW = 3.0  # plot/out-of-transit window, in transit durations either side
EPHEMERIS_COLS = ["koi_period", "koi_time0bk", "koi_duration"]


# ---------- Per-star preparation ----------

def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_PATH)


def star_ephemerides(kepid: int, catalog: pd.DataFrame) -> dict:
    """{kepoi_name: (period_days, t0_bkjd, duration_days)} for every usable KOI on this star."""
    rows = catalog[catalog["kepid"] == kepid].dropna(subset=EPHEMERIS_COLS)
    rows = rows[(rows["koi_period"] > 0) & (rows["koi_duration"] > 0)]
    return {r.kepoi_name: (r.koi_period, r.koi_time0bk, r.koi_duration / 24) for r in rows.itertuples()}


def prepare_star(raw, ephemerides: dict):
    """Clean once per star, keeping every known KOI's transits out of the trend fit."""
    flat, _, _ = clean(raw, ephemeris=list(ephemerides.values()))
    t, y = flat.time.value, flat.flux.value
    dy = flat.flux_err.value
    dy = np.where(np.isfinite(dy), dy, np.nanmedian(dy))
    return t, y, dy


def koi_data(t, y, dy, ephemerides: dict, name: str):
    """Data for one KOI with every *other* KOI's transits (+/- 1 duration) removed."""
    keep = np.ones(len(t), dtype=bool)
    for other, (period, t0, duration) in ephemerides.items():
        if other != name:
            keep &= np.abs(fold_hours(t, period, t0)) >= duration * 24
    return t[keep], y[keep], dy[keep]


# ---------- Features ----------

def phase_in_durations(t, period, t0, duration_days):
    """Time -> distance from nearest mid-transit, in units of the transit duration."""
    return fold_hours(t, period, t0) / (duration_days * 24)


def shape_ratio(t, y, period, t0, duration_days) -> float:
    """Mean depth of the transit's outer edges divided by mean depth at its center."""
    x = np.abs(phase_in_durations(t, period, t0, duration_days))
    center, edges, outside = x < 1 / 6, (x >= 1 / 6) & (x < 0.5), (x > 1) & (x < WINDOW)
    if center.sum() < 5 or edges.sum() < 5 or outside.sum() < 20:
        return np.nan
    baseline = np.median(y[outside])
    depth_center = baseline - np.median(y[center])
    depth_edges = baseline - np.median(y[edges])
    return depth_edges / depth_center if depth_center > 0 else np.nan


def ratio(a, b):
    return a / b if b > 0 else np.nan


def extract_features(t, y, dy, period, t0, duration_days) -> dict:
    bls = BoxLeastSquares(t, y, dy)
    s = bls.compute_stats(period, duration_days, t0)

    depth, depth_err = s["depth"]
    odd, odd_err = s["depth_odd"]
    even, even_err = s["depth_even"]
    secondary, secondary_err = s["depth_phased"]

    ll = np.asarray(s["per_transit_log_likelihood"])
    ll_total = ll[ll > 0].sum()
    out_of_transit = np.abs(phase_in_durations(t, period, t0, duration_days)) > 1

    return {
        "period_days": period,
        "duration_hours": duration_days * 24,
        "duration_frac": duration_days / period,
        "depth_ppm": depth * 1e6,
        "depth_snr": ratio(depth, depth_err),
        "odd_even_sigma": ratio(abs(odd - even), np.hypot(odd_err, even_err)),
        "odd_even_rel": ratio(abs(odd - even), depth),
        "secondary_ppm": secondary * 1e6,
        "secondary_snr": ratio(secondary, secondary_err),
        "secondary_ratio": ratio(secondary, depth),
        "shape_ratio": shape_ratio(t, y, period, t0, duration_days),
        "harmonic_dll": float(s["harmonic_delta_log_likelihood"]),
        "max_single_frac": ratio(ll.max(), ll_total) if len(ll) else np.nan,
        "n_transits": int(np.sum(np.asarray(s["per_transit_count"]) > 0)),
        "noise_ppm": float(np.std(y[out_of_transit]) * 1e6),
    }


# ---------- Vetting plot ----------

def plot_vetting(t, y, period, t0, duration_days, title: str, path) -> None:
    fig = vetting_figure(t, y, period, t0, duration_days, title)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def vetting_figure(t, y, period, t0, duration_days, title: str) -> plt.Figure:
    """Three panels: full transit, odd vs. even transits, and the phase-0.5 secondary window."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    flux = (y - 1.0) * 1e6
    dur_h = duration_days * 24
    width = dur_h / 5

    def window(phase_h):
        return np.abs(phase_h) < WINDOW * dur_h

    # 1. Primary transit
    ph = fold_hours(t, period, t0)
    w = window(ph)
    axes[0].scatter(ph[w], flux[w], s=3, color=POINT_COLOR)
    bx, by = binned_median(ph[w], flux[w], width)
    axes[0].plot(bx, by, color=DATA_COLOR, linewidth=2)
    style(axes[0], "Primary transit")

    # 2. Odd vs. even transits
    epoch = np.round((t - t0) / period).astype(int)
    for parity, color, label in ((1, DATA_COLOR, "Odd transits"), (0, TREND_COLOR, "Even transits")):
        sel = w & (epoch % 2 == parity)
        if sel.sum() > 10:
            bx, by = binned_median(ph[sel], flux[sel], width)
            axes[1].plot(bx, by, color=color, linewidth=2, label=label)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    style(axes[1], "Odd vs. even")

    # 3. Secondary eclipse window (half an orbit after transit)
    ph2 = fold_hours(t, period, t0 + period / 2)
    w2 = window(ph2)
    axes[2].scatter(ph2[w2], flux[w2], s=3, color=POINT_COLOR)
    bx, by = binned_median(ph2[w2], flux[w2], width)
    axes[2].plot(bx, by, color=DATA_COLOR, linewidth=2)
    style(axes[2], "Phase 0.5 (secondary eclipse?)")

    # Share y-limits so a secondary dip is judged on the same scale as the primary
    low, high = np.nanpercentile(flux[w], [0.5, 99.5])
    for ax in axes:
        ax.set_ylim(low - 0.1 * (high - low), high + 0.1 * (high - low))
        ax.set_xlabel("Hours from center", color=INK, fontsize=9)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute vetting features for one KOI.")
    parser.add_argument("--koi", default="K00072.01", help="KOI name, e.g. K00072.01")
    parser.add_argument("--quarters", type=int, default=None, help="Only use the first N quarters")
    args = parser.parse_args()

    catalog = load_catalog()
    match = catalog[catalog["kepoi_name"] == args.koi]
    if match.empty:
        raise SystemExit(f"{args.koi} not found in {CATALOG_PATH}. Run build_koi_list.py first.")
    koi = match.iloc[0]

    ephemerides = star_ephemerides(koi["kepid"], catalog)
    period, t0, duration = ephemerides[args.koi]
    name = koi["kepler_name"] if isinstance(koi["kepler_name"], str) else args.koi
    print(f"{args.koi} ({name}) on KIC {koi['kepid']}: {koi['koi_disposition']}")
    print(f"Catalog: period {period:.5f} d, duration {duration * 24:.2f} h, depth {koi['koi_depth']:.0f} ppm")
    others = [k for k in ephemerides if k != args.koi]
    if others:
        print(f"Other KOIs on this star (masked out): {', '.join(others)}")

    raw = fetch(f"KIC {koi['kepid']}", args.quarters)
    t, y, dy = koi_data(*prepare_star(raw, ephemerides), ephemerides, args.koi)

    features = extract_features(t, y, dy, period, t0, duration)
    print("\n--- Features ---")
    for key, value in features.items():
        print(f"  {key:<16} {value:>12.4g}")

    OUT_DIR.mkdir(exist_ok=True)
    title = f"{args.koi} ({name}): {koi['koi_disposition']}, P = {period:.4f} d"
    plot_vetting(t, y, period, t0, duration, title, OUT_DIR / f"{args.koi}_vetting.png")


if __name__ == "__main__":
    main()
