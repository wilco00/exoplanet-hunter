"""Milestone 4c: compute features for every KOI in data/training_set.csv.

For each star: download its light curve once, clean it once, then compute features for
each of its training KOIs. Results are appended to data/features.csv after every star,
so you can stop at any time (Ctrl+C) and re-run to pick up where it left off.

Downloads go to data/lightkurve_cache/ (on this drive) instead of your home folder.

Usage:
    python build_features.py --limit 10      # try 10 stars first to check timing
    python build_features.py                 # everything (resumes automatically)
    python build_features.py --retry-failed  # give previously failed KOIs another try
"""
import argparse
import contextlib
import io
import time
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd  # noqa: E402

from build_koi_list import DATA_DIR, TRAINING_PATH  # noqa: E402
from extract_features import extract_features, koi_data, load_catalog, prepare_star, star_ephemerides  # noqa: E402
from fetch_lightcurve import fetch  # noqa: E402

FEATURES_PATH = DATA_DIR / "features.csv"
FAILED_PATH = DATA_DIR / "failed.csv"
CACHE_DIR = DATA_DIR / "lightkurve_cache"
MIN_TRANSITS = 2


def append_rows(path, rows: list) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def names_in(path) -> set:
    return set(pd.read_csv(path)["kepoi_name"]) if path.exists() else set()


def process_star(kepid: int, targets: pd.DataFrame, catalog: pd.DataFrame):
    """Return (feature rows, failure rows) for one star's training KOIs."""
    with contextlib.redirect_stdout(io.StringIO()):  # keep fetch()'s progress chatter quiet
        raw = fetch(f"KIC {kepid}", None, download_dir=str(CACHE_DIR))

    ephemerides = star_ephemerides(kepid, catalog)
    t, y, dy = prepare_star(raw, ephemerides)

    rows, failures = [], []
    for koi in targets.itertuples():
        period, t0, duration = ephemerides[koi.kepoi_name]
        features = extract_features(*koi_data(t, y, dy, ephemerides, koi.kepoi_name), period, t0, duration)
        base = {"kepid": kepid, "kepoi_name": koi.kepoi_name}
        if features["n_transits"] < MIN_TRANSITS:
            failures.append({**base, "error": f"only {features['n_transits']} transit(s) in data"})
        else:
            rows.append({**base, "label": koi.label, "koi_disposition": koi.koi_disposition, **features})
    return rows, failures


def summarize() -> None:
    if not FEATURES_PATH.exists():
        return
    df = pd.read_csv(FEATURES_PATH)
    print(f"\n--- {FEATURES_PATH}: {len(df)} KOIs "
          f"({(df['label'] == 1).sum()} confirmed, {(df['label'] == 0).sum()} false positive) ---")
    cols = ["depth_snr", "odd_even_rel", "secondary_ratio", "shape_ratio", "duration_frac", "max_single_frac"]
    print("Median by class:")
    print(df.groupby("koi_disposition")[cols].median().round(3).to_string())
    if FAILED_PATH.exists():
        print(f"\nFailures logged: {len(pd.read_csv(FAILED_PATH))} (see {FAILED_PATH})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute features for the whole training set.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many stars")
    parser.add_argument("--retry-failed", action="store_true", help="Retry KOIs listed in failed.csv")
    args = parser.parse_args()

    training = pd.read_csv(TRAINING_PATH)
    catalog = load_catalog()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    done = names_in(FEATURES_PATH)
    if args.retry_failed and FAILED_PATH.exists():
        FAILED_PATH.unlink()
    skip = done | names_in(FAILED_PATH)

    todo = training[~training["kepoi_name"].isin(skip)]
    stars = list(todo.groupby("kepid"))
    if args.limit:
        stars = stars[: args.limit]
    print(f"{len(done)} KOIs already done; processing {len(stars)} star(s) "
          f"({sum(len(g) for _, g in stars)} KOIs)")

    start = time.perf_counter()
    try:
        for i, (kepid, targets) in enumerate(stars, 1):
            t_star = time.perf_counter()
            try:
                rows, failures = process_star(kepid, targets, catalog)
            except Exception as e:  # noqa: BLE001 -- log and keep going
                rows = []
                failures = [{"kepid": kepid, "kepoi_name": n, "error": f"{type(e).__name__}: {e}"}
                            for n in targets["kepoi_name"]]
            append_rows(FEATURES_PATH, rows)
            append_rows(FAILED_PATH, failures)

            elapsed = time.perf_counter() - start
            eta_min = elapsed / i * (len(stars) - i) / 60
            status = f"{len(rows)} ok" + (f", {len(failures)} failed" if failures else "")
            print(f"[{i}/{len(stars)}] KIC {kepid}: {status} "
                  f"({time.perf_counter() - t_star:.0f} s; ~{eta_min:.0f} min left)")
    except KeyboardInterrupt:
        print("\nStopped. Everything finished so far is saved; re-run to resume.")

    summarize()


if __name__ == "__main__":
    main()
