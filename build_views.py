"""Milestone 6a: turn each KOI's light curve into fixed-size "views" for the CNN.

A neural network needs every input to be the same size, but light curves have different
lengths, periods and depths. Following Google's AstroNet (Shallue & Vanderburg 2018),
each KOI becomes two 1D "pictures":

  global view  the whole orbit, folded on the period, binned to 2001 points.
               Shows secondary eclipses and anything else happening around the orbit.
  local view   +/- 2 transit durations around the transit, binned to 201 points.
               Shows the transit's shape (U vs. V) in detail.

Each view is normalized so its median is 0 and its lowest point is -1, so the network
learns from shapes rather than raw depths.

Light curves are read straight from data/lightkurve_cache/ (downloaded in milestone 4c),
so nothing is re-downloaded. Each KOI's views are saved to data/views/<KOI>.npz as they're
made (stop any time with Ctrl+C and re-run to resume), then everything is bundled into
data/views.npz.

Usage:
    python build_views.py --limit 5     # quick check on 5 stars
    python build_views.py               # all KOIs in data/features.csv
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import lightkurve as lk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_features import CACHE_DIR, FEATURES_PATH  # noqa: E402
from build_koi_list import DATA_DIR  # noqa: E402
from extract_features import koi_data, load_catalog, prepare_star, star_ephemerides  # noqa: E402

VIEWS_DIR = DATA_DIR / "views"
VIEWS_PATH = DATA_DIR / "views.npz"
PLOT_PATH = DATA_DIR.parent / "plots" / "view_examples.png"
GLOBAL_BINS, LOCAL_BINS = 2001, 201
LOCAL_HALF_WIDTH = 2.0  # local view spans +/- this many transit durations

# Plot colors: dataviz reference palette slots 1-2 + neutral ink
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"


# ---------- Loading ----------

def load_cached(kepid: int) -> lk.LightCurve:
    """Read and stitch a star's already-downloaded Kepler long-cadence files."""
    star_dir = CACHE_DIR / "mastDownload" / "Kepler"
    files = sorted(star_dir.glob(f"kplr{int(kepid):09d}_lc_*/*_llc.fits"))
    if not files:
        raise FileNotFoundError(f"No cached light curve for KIC {kepid}")
    return lk.LightCurveCollection([lk.read(f) for f in files]).stitch().remove_nans()


# ---------- Views ----------

def binned_median(x: np.ndarray, y: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    """Median of y in n_bins equal bins over [lo, hi); empty bins filled by interpolation."""
    idx = np.floor((x - lo) / (hi - lo) * n_bins).astype(int)
    inside = (idx >= 0) & (idx < n_bins)
    medians = pd.Series(y[inside]).groupby(idx[inside]).median().reindex(range(n_bins)).to_numpy()
    filled = np.isfinite(medians)
    if filled.sum() < 2:
        raise ValueError("too few data points to build a view")
    positions = np.arange(n_bins)
    return np.interp(positions, positions[filled], medians[filled])


def normalize(view: np.ndarray) -> np.ndarray:
    """Shift so the median is 0, scale so the lowest point is -1."""
    view = view - np.median(view)
    low = view.min()
    return view / -low if low < 0 else view


def make_views(t, y, period, t0, duration_days):
    phase = (t - t0 + 0.5 * period) % period - 0.5 * period  # days from mid-transit
    global_view = binned_median(phase, y, -period / 2, period / 2, GLOBAL_BINS)
    half = min(LOCAL_HALF_WIDTH * duration_days, period / 2)
    local_view = binned_median(phase, y, -half, half, LOCAL_BINS)
    return normalize(global_view), normalize(local_view)


# ---------- Bundling + example plot ----------

def bundle(table: pd.DataFrame) -> None:
    rows, globals_, locals_ = [], [], []
    for r in table.itertuples():
        path = VIEWS_DIR / f"{r.kepoi_name}.npz"
        if path.exists():
            v = np.load(path)
            rows.append(r)
            globals_.append(v["global_view"])
            locals_.append(v["local_view"])
    np.savez_compressed(
        VIEWS_PATH,
        kepoi_name=np.array([r.kepoi_name for r in rows]),
        kepid=np.array([r.kepid for r in rows]),
        label=np.array([r.label for r in rows]),
        global_view=np.array(globals_, dtype=np.float32),
        local_view=np.array(locals_, dtype=np.float32),
    )
    print(f"\nBundled {len(rows)} KOIs into {VIEWS_PATH}")


def plot_examples(table: pd.DataFrame) -> None:
    """Two strongest planets and two strongest false positives, global + local views."""
    have = table[table["kepoi_name"].map(lambda n: (VIEWS_DIR / f"{n}.npz").exists())]
    picks = pd.concat([
        have[have["label"] == 1].nlargest(2, "depth_snr"),
        have[have["label"] == 0].nlargest(2, "depth_snr"),
    ])
    if picks.empty:
        return
    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 2.4 * len(picks)), facecolor=SURFACE,
                             squeeze=False, gridspec_kw={"width_ratios": [2, 1]})
    for row, r in zip(axes, picks.itertuples()):
        v = np.load(VIEWS_DIR / f"{r.kepoi_name}.npz")
        color = BLUE if r.label == 1 else ORANGE
        kind = "planet" if r.label == 1 else "false positive"
        for ax, key, name in ((row[0], "global_view", "global"), (row[1], "local_view", "local")):
            ax.plot(v[key], color=color, linewidth=1.2)
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK, labelsize=8)
            ax.set_title(f"{r.kepoi_name} ({kind}): {name} view", loc="left", color=INK, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Bin", color=INK, fontsize=9)
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build CNN input views for every KOI with features.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many stars")
    args = parser.parse_args()

    table = pd.read_csv(FEATURES_PATH)
    catalog = load_catalog()
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)

    todo = table[~table["kepoi_name"].map(lambda n: (VIEWS_DIR / f"{n}.npz").exists())]
    stars = list(todo.groupby("kepid"))
    if args.limit:
        stars = stars[: args.limit]
    print(f"{len(table) - len(todo)} KOIs already have views; processing {len(stars)} star(s)")

    start, failures = time.perf_counter(), 0
    try:
        for i, (kepid, targets) in enumerate(stars, 1):
            try:
                ephemerides = star_ephemerides(kepid, catalog)
                t, y, dy = prepare_star(load_cached(kepid), ephemerides)
                for name in targets["kepoi_name"]:
                    period, t0, duration = ephemerides[name]
                    tk, yk, _ = koi_data(t, y, dy, ephemerides, name)
                    g, l = make_views(tk, yk, period, t0, duration)
                    np.savez(VIEWS_DIR / f"{name}.npz", global_view=g, local_view=l)
                status = "ok"
            except Exception as e:  # noqa: BLE001 -- report and keep going
                failures += 1
                status = f"FAILED ({type(e).__name__}: {e})"
            elapsed = time.perf_counter() - start
            eta_min = elapsed / i * (len(stars) - i) / 60
            print(f"[{i}/{len(stars)}] KIC {kepid}: {status} (~{eta_min:.0f} min left)")
    except KeyboardInterrupt:
        print("\nStopped. Finished views are saved; re-run to resume.")

    if failures:
        print(f"\n{failures} star(s) failed (listed above).")
    bundle(table)
    plot_examples(table)


if __name__ == "__main__":
    main()
