"""Milestone 2: clean a Kepler light curve so transit dips stand out.

Steps:
  1. Clip HIGH outliers only (cosmic rays, flares). Low points are kept,
     because transits are low points.
  2. Flatten: fit a smooth trend to the star's slow variability and divide it out.
  3. Report noise before and after, and plot raw vs. cleaned.

Usage:
    python clean_lightcurve.py --quarters 2
    python clean_lightcurve.py --quarters 2 --window 301   # try a different trend window
"""
import argparse
import warnings

# lightkurve warns about an optional dependency we don't use; silence it before import
warnings.filterwarnings("ignore", message=".*oktopus.*")

import lightkurve as lk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fetch_lightcurve import DEFAULT_TARGET, OUT_DIR, fetch  # noqa: E402

DEFAULT_WINDOW = 101  # cadences; 101 x 30 min ~ 2 days
ZOOM_DAYS = 5
GAP_DAYS = 0.5  # don't draw trend lines across data gaps longer than this

# Colors: categorical slots 1-2 of the dataviz reference palette, plus neutral ink
DATA_COLOR = "#2a78d6"
TREND_COLOR = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def clean(lc: lk.LightCurve, window_length: int = DEFAULT_WINDOW, sigma_upper: float = 4.0,
          ephemeris: tuple | None = None):
    """Return (flattened light curve, fitted trend, number of clipped points).

    window_length is in cadences and must be odd. It should be several times longer
    than the longest transit you care about (Kepler-10c's lasts ~7 h = ~14 cadences),
    or the trend fit starts eating into the dips.

    ephemeris: optional (period_days, mid_transit_bkjd, duration_days) of a known signal,
    or a list of them for a star with several. Those transits (padded to 2x duration)
    are left out of the trend fit, so the fit can't bend into them and make them shallower.
    """
    if window_length % 2 == 0:
        window_length += 1

    clipped, mask = lc.remove_outliers(sigma_lower=np.inf, sigma_upper=sigma_upper, return_mask=True)

    transit_mask = None
    if ephemeris:
        ephemerides = [ephemeris] if np.isscalar(ephemeris[0]) else ephemeris
        transit_mask = np.zeros(len(clipped), dtype=bool)
        for period, t0, duration in ephemerides:
            transit_mask |= np.asarray(
                clipped.create_transit_mask(period=period, transit_time=t0, duration=2 * duration))

    flat, trend = clipped.flatten(window_length=window_length, return_trend=True, mask=transit_mask)
    return flat, trend, int(mask.sum())


def ppm(lc: lk.LightCurve) -> np.ndarray:
    """Normalized flux (~1.0) -> parts-per-million offset from 1."""
    return (lc.flux.value - 1.0) * 1e6


def with_gaps(t: np.ndarray, y: np.ndarray, max_gap: float = GAP_DAYS):
    """Insert NaNs at data gaps so a line plot breaks instead of bridging them."""
    idx = np.where(np.diff(t) > max_gap)[0] + 1
    return np.insert(t, idx, np.nan), np.insert(y, idx, np.nan)


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_ylabel("Relative flux (ppm)", color=INK, fontsize=9)
    ax.set_xlabel("Time (BKJD days)", color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def plot(raw: lk.LightCurve, trend: lk.LightCurve, flat: lk.LightCurve, target: str, path) -> None:
    fig, (ax_raw, ax_flat, ax_zoom) = plt.subplots(3, 1, figsize=(11, 10), facecolor=SURFACE)

    # 1. Raw flux with the fitted trend on top
    ax_raw.scatter(raw.time.value, ppm(raw), s=2, color=DATA_COLOR, label="Kepler flux")
    tt, ty = with_gaps(trend.time.value, ppm(trend))
    ax_raw.plot(tt, ty, color=TREND_COLOR, linewidth=2, label="Fitted trend")
    ax_raw.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    style(ax_raw, f"{target}: raw flux and fitted trend")

    # 2. Flattened (raw / trend)
    ax_flat.scatter(flat.time.value, ppm(flat), s=2, color=DATA_COLOR)
    ax_flat.sharex(ax_raw)
    style(ax_flat, "After flattening")

    # 3. Zoom on the first few days of flattened data
    t0 = flat.time.value[0]
    zoom = flat.truncate(t0, t0 + ZOOM_DAYS)
    ax_zoom.scatter(zoom.time.value, ppm(zoom), s=8, color=DATA_COLOR)
    style(ax_zoom, f"After flattening: first {ZOOM_DAYS} days")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean (detrend) a Kepler light curve.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Star name or KIC ID")
    parser.add_argument("--quarters", type=int, default=None, help="Only use the first N quarters")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Trend window in cadences (odd)")
    args = parser.parse_args()

    raw = fetch(args.target, args.quarters)
    flat, trend, n_clipped = clean(raw, window_length=args.window)

    print("\n--- Cleaning summary ---")
    print(f"Clipped high outliers: {n_clipped}")
    print(f"Scatter before:        {np.std(ppm(raw)):.0f} ppm")
    print(f"Scatter after:         {np.std(ppm(flat)):.0f} ppm")
    print(f"CDPP (6.5 h):          {flat.estimate_cdpp().value:.1f} ppm")

    OUT_DIR.mkdir(exist_ok=True)
    slug = args.target.replace(" ", "_")
    plot(raw, trend, flat, args.target, OUT_DIR / f"{slug}_clean.png")


if __name__ == "__main__":
    main()
