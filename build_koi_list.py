"""Milestone 4a: build the labeled list of Kepler Objects of Interest (KOIs) to train on.

Pulls NASA Exoplanet Archive's cumulative KOI table, then:
  - CONFIRMED planets      -> label 1
  - FALSE POSITIVE signals -> label 0
  - CANDIDATE (unresolved) -> held out entirely, saved separately for milestone 8

Each row already has the catalog period, mid-transit time and duration, so later
steps can fold each star's light curve without running BLS.

Usage:
    python build_koi_list.py                 # ~1,000 balanced KOIs
    python build_koi_list.py --size 300
    python build_koi_list.py --refresh       # re-download the catalog
"""
import argparse
import io
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "koi_cumulative.csv"
TRAINING_PATH = DATA_DIR / "training_set.csv"
CANDIDATES_PATH = DATA_DIR / "candidates.csv"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
COLUMNS = [
    "kepid",            # Kepler Input Catalog star ID
    "kepoi_name",       # e.g. K00072.01
    "kepler_name",      # e.g. Kepler-10 b (confirmed planets only)
    "koi_disposition",  # CONFIRMED / FALSE POSITIVE / CANDIDATE
    "koi_period",       # days
    "koi_time0bk",      # mid-transit time, BKJD (same clock as lightkurve's Kepler data)
    "koi_duration",     # hours
    "koi_depth",        # ppm
    "koi_model_snr",    # transit signal-to-noise from NASA's pipeline
]
MAX_DURATION_HOURS = 15  # longer transits get partly absorbed by the ~2-day flattening window
SEED = 42


def download_catalog() -> pd.DataFrame:
    query = f"select {','.join(COLUMNS)} from cumulative"
    url = f"{TAP_URL}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"
    print("Downloading KOI catalog from NASA Exoplanet Archive...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return pd.read_csv(io.StringIO(resp.read().decode("utf-8")))


def load_catalog(refresh: bool) -> pd.DataFrame:
    if CATALOG_PATH.exists() and not refresh:
        print(f"Using cached catalog: {CATALOG_PATH}")
        return pd.read_csv(CATALOG_PATH)
    df = download_catalog()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(CATALOG_PATH, index=False)
    print(f"Saved {CATALOG_PATH}")
    return df


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows we can't fold reliably."""
    has_ephemeris = df[["koi_period", "koi_time0bk", "koi_duration"]].notna().all(axis=1)
    sane = (df["koi_period"] > 0) & (df["koi_duration"] > 0) & (df["koi_duration"] <= MAX_DURATION_HOURS)
    return df[has_ephemeris & sane]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the labeled KOI training list.")
    parser.add_argument("--size", type=int, default=1000, help="Total KOIs (split evenly between classes)")
    parser.add_argument("--refresh", action="store_true", help="Re-download the catalog")
    args = parser.parse_args()

    catalog = load_catalog(args.refresh)
    print(f"\nCatalog: {len(catalog):,} KOIs")
    print(catalog["koi_disposition"].value_counts().to_string())

    clean = usable(catalog)
    print(f"\nUsable after ephemeris/duration checks: {len(clean):,}")

    per_class = args.size // 2
    confirmed = clean[clean["koi_disposition"] == "CONFIRMED"]
    false_pos = clean[clean["koi_disposition"] == "FALSE POSITIVE"]
    per_class = min(per_class, len(confirmed), len(false_pos))

    training = pd.concat([
        confirmed.sample(per_class, random_state=SEED).assign(label=1),
        false_pos.sample(per_class, random_state=SEED).assign(label=0),
    ]).sort_values("kepid").reset_index(drop=True)

    candidates = clean[clean["koi_disposition"] == "CANDIDATE"].reset_index(drop=True)

    training.to_csv(TRAINING_PATH, index=False)
    candidates.to_csv(CANDIDATES_PATH, index=False)

    print("\n--- Training set ---")
    print(f"KOIs:          {len(training):,}  ({per_class} confirmed, {per_class} false positive)")
    print(f"Unique stars:  {training['kepid'].nunique():,}  (= light curves to download in 4c)")
    print("\nMedian values by class:")
    summary = training.groupby("koi_disposition")[["koi_period", "koi_duration", "koi_depth", "koi_model_snr"]].median()
    print(summary.round(2).to_string())
    print(f"\nSaved {TRAINING_PATH}")
    print(f"Saved {CANDIDATES_PATH} ({len(candidates):,} held-out candidates for milestone 8)")


if __name__ == "__main__":
    main()
