"""Milestone 1: download a Kepler light curve from NASA's MAST archive and plot it.

Usage:
    python fetch_lightcurve.py                     # Kepler-10, all quarters
    python fetch_lightcurve.py --quarters 2        # first 2 quarters only (faster)
    python fetch_lightcurve.py --target "KIC 11904151"
"""
import argparse
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TARGET = "Kepler-10"  # KIC 11904151, host of Kepler-10b (period ~0.84 days)
OUT_DIR = Path("plots")
ZOOM_DAYS = 5


def fetch(target: str, max_quarters: int | None, download_dir: str | None = None) -> lk.LightCurve:
    """Search MAST for Kepler long-cadence data, download it, and stitch quarters together.

    download_dir: where to cache the FITS files (default: lightkurve's cache in your home folder).
    """
    print(f"Searching MAST for {target}...")
    results = lk.search_lightcurve(target, mission="Kepler", author="Kepler", exptime=1800)
    if len(results) == 0:
        raise LookupError(f"No Kepler long-cadence light curves found for '{target}'.")

    print(f"Found {len(results)} quarter(s) of data.")
    if max_quarters:
        results = results[:max_quarters]

    print(f"Downloading {len(results)} quarter(s) (cached after the first run)...")
    collection = results.download_all(download_dir=download_dir)
    if collection is None or len(collection) == 0:
        raise RuntimeError(f"Download failed for '{target}'.")

    # stitch() normalizes each quarter to ~1.0, then joins them into one light curve
    return collection.stitch().remove_nans()


def save_plot(lc: lk.LightCurve, title: str, path: Path) -> None:
    ax = lc.scatter(s=1)
    ax.set_title(title)
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and plot a Kepler light curve.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Star name or KIC ID")
    parser.add_argument("--quarters", type=int, default=None, help="Only download the first N quarters")
    args = parser.parse_args()

    lc = fetch(args.target, args.quarters)

    time = lc.time.value
    flux = lc.flux.value
    print("\n--- Summary ---")
    print(f"Data points: {len(lc):,}")
    print(f"Time span:   {time[-1] - time[0]:.1f} days")
    print(f"Flux scatter: {np.nanstd(flux) * 1e6:.0f} ppm")

    OUT_DIR.mkdir(exist_ok=True)
    slug = args.target.replace(" ", "_")

    save_plot(lc, f"{args.target}: full light curve", OUT_DIR / f"{slug}_full.png")

    zoom = lc.truncate(time[0], time[0] + ZOOM_DAYS)
    save_plot(zoom, f"{args.target}: first {ZOOM_DAYS} days", OUT_DIR / f"{slug}_zoom.png")


if __name__ == "__main__":
    main()
