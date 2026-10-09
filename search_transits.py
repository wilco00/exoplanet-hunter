"""Milestone 3: search a cleaned light curve for periodic transits with Box Least Squares (BLS).

BLS tries every trial period and duration and asks: "if the star dimmed by a constant
amount for this long, once every P days, how much better would that fit the data than
a flat line?" The period with the biggest improvement is the transit candidate.

After a signal is found, its transits are masked out and the search runs again, so a
second (weaker or longer-period) planet can show up.

Usage:
    python search_transits.py --quarters 2        # quick: should recover Kepler-10b
    python search_transits.py --planets 2         # all quarters: Kepler-10b, then 10c
"""
import argparse
import time as timer
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.timeseries import BoxLeastSquares  # noqa: E402

from clean_lightcurve import DATA_COLOR, INK, SURFACE, TREND_COLOR, clean, style  # noqa: E402
from fetch_lightcurve import DEFAULT_TARGET, OUT_DIR, fetch  # noqa: E402

# Trial transit durations in days (~1 h to ~7 h covers most Kepler planets)
DURATIONS = np.array([0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22, 0.29])
MIN_PERIOD = 0.5        # days; must be longer than the longest trial duration
DEFAULT_MAX_PERIOD = 50.0
OVERSAMPLE = 2          # period-grid density: higher = finer grid, slower search
POINT_COLOR = "#b5b3ad"  # neutral gray for raw folded points


def period_grid(baseline: float, min_period: float, max_period: float) -> np.ndarray:
    """Log-spaced trial periods.

    If the trial period is off by dP, the predicted transit drifts by (baseline / P) * dP
    over the whole dataset. Keeping that drift under (shortest duration / OVERSAMPLE)
    means no real transit can fall between grid points.
    """
    step = DURATIONS.min() / (baseline * OVERSAMPLE)  # max fractional period step
    n = int(np.ceil(np.log(max_period / min_period) / np.log1p(step)))
    return np.geomspace(min_period, max_period, n)


def search(t: np.ndarray, y: np.ndarray, dy: np.ndarray, max_period: float) -> dict:
    """Run one BLS search and return the strongest signal."""
    baseline = t.max() - t.min()
    max_period = min(max_period, baseline / 2)  # need at least 2 transits in the data
    periods = period_grid(baseline, MIN_PERIOD, max_period)

    print(f"  Testing {len(periods):,} trial periods ({MIN_PERIOD}-{max_period:.1f} days)...")
    start = timer.perf_counter()
    bls = BoxLeastSquares(t, y, dy)
    result = bls.power(periods, DURATIONS)
    print(f"  Done in {timer.perf_counter() - start:.1f} s")

    i = int(np.argmax(result.power))
    power = result.power
    period, duration, t0 = result.period[i], result.duration[i], result.transit_time[i]
    stats = bls.compute_stats(period, duration, t0)

    return {
        "bls": bls,
        "t": t,
        "y": y,
        "periods": periods,
        "power": power,
        "period": period,
        "t0": t0,
        "duration": duration,
        "depth_ppm": result.depth[i] * 1e6,
        "depth_snr": result.depth_snr[i],
        # Signal Detection Efficiency: how far the peak stands above the typical power
        "sde": (power[i] - power.mean()) / power.std(),
        "n_transits": int(np.sum(stats["per_transit_count"] > 0)),
    }


def report(n: int, r: dict) -> None:
    print(f"\n  Signal {n}")
    print(f"    Period:        {r['period']:.5f} days")
    print(f"    Mid-transit:   BKJD {r['t0']:.4f}")
    print(f"    Duration:      {r['duration'] * 24:.1f} h")
    print(f"    Depth:         {r['depth_ppm']:.0f} ppm  (depth SNR {r['depth_snr']:.1f})")
    print(f"    SDE:           {r['sde']:.1f}")
    print(f"    Transits seen: {r['n_transits']}")


def fold_hours(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time -> hours from the nearest mid-transit."""
    return ((t - t0 + 0.5 * period) % period - 0.5 * period) * 24


def binned_median(x: np.ndarray, y: np.ndarray, width: float):
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges)
    centers, medians = [], []
    for k in range(1, len(edges)):
        in_bin = idx == k
        if in_bin.sum() >= 3:
            centers.append(0.5 * (edges[k - 1] + edges[k]))
            medians.append(np.median(y[in_bin]))
    return np.array(centers), np.array(medians)


def plot(found: list, target: str, path) -> None:
    rows = len(found)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows), facecolor=SURFACE, squeeze=False)

    for row, r in enumerate(found):
        ax_pg, ax_fold = axes[row]

        # Left: periodogram (BLS power vs. trial period)
        ax_pg.plot(r["periods"], r["power"], color=DATA_COLOR, linewidth=0.8)
        ax_pg.axvline(r["period"], color=TREND_COLOR, linewidth=1.5, linestyle="--",
                      label=f"Best period: {r['period']:.4f} d")
        ax_pg.set_xscale("log")
        style(ax_pg, f"Signal {row + 1}: BLS periodogram")
        ax_pg.set_xlabel("Trial period (days)", color=INK, fontsize=9)
        ax_pg.set_ylabel("BLS power", color=INK, fontsize=9)
        ax_pg.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)

        # Right: light curve folded on the best period, zoomed on the transit
        half_window = 3 * r["duration"] * 24  # hours either side of mid-transit
        phase = fold_hours(r["t"], r["period"], r["t0"])
        near = np.abs(phase) < half_window
        flux_ppm = (r["y"][near] - 1.0) * 1e6

        ax_fold.scatter(phase[near], flux_ppm, s=3, color=POINT_COLOR, label="Folded data")
        bx, by = binned_median(phase[near], flux_ppm, width=r["duration"] * 24 / 5)
        ax_fold.plot(bx, by, color=DATA_COLOR, linewidth=2, label="Binned median")

        model_h = np.linspace(-half_window, half_window, 1000)
        model = r["bls"].model(r["t0"] + model_h / 24, r["period"], r["duration"], r["t0"])
        ax_fold.plot(model_h, (model - 1.0) * 1e6, color=TREND_COLOR, linewidth=1.5, label="BLS box model")

        style(ax_fold, f"{target}: folded on {r['period']:.4f} d")
        ax_fold.set_xlabel("Hours from mid-transit", color=INK, fontsize=9)
        ax_fold.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a Kepler light curve for transits with BLS.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Star name or KIC ID")
    parser.add_argument("--quarters", type=int, default=None, help="Only use the first N quarters")
    parser.add_argument("--planets", type=int, default=1, help="How many signals to search for")
    parser.add_argument("--max-period", type=float, default=DEFAULT_MAX_PERIOD, help="Longest trial period (days)")
    args = parser.parse_args()

    raw = fetch(args.target, args.quarters)
    flat, _, _ = clean(raw)

    t = flat.time.value
    y = flat.flux.value
    dy = flat.flux_err.value
    dy = np.where(np.isfinite(dy), dy, np.nanmedian(dy))

    found = []
    for n in range(1, args.planets + 1):
        print(f"\nSearch {n} ({len(t):,} points)")
        r = search(t, y, dy, args.max_period)
        report(n, r)
        found.append(r)

        # Mask this signal's transits (2x duration for margin) before the next search
        in_transit = r["bls"].transit_mask(t, r["period"], 2 * r["duration"], r["t0"])
        t, y, dy = t[~in_transit], y[~in_transit], dy[~in_transit]

    OUT_DIR.mkdir(exist_ok=True)
    slug = args.target.replace(" ", "_")
    plot(found, args.target, OUT_DIR / f"{slug}_bls.png")


if __name__ == "__main__":
    main()
