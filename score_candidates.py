"""Milestone 8: score every held-out CANDIDATE KOI with both models ("what my model flagged").

Each candidate goes through inference.analyze(), the same parity-checked path the dashboard
uses: download (into data/lightkurve_cache/) -> clean -> features + views -> classic, CNN,
ensemble. Results are appended after every star, so the run can be stopped (Ctrl+C) and
re-run to resume. Stars are processed in a fixed random order, so a partial run is still a
fair random sample.

astroquery's file downloads have no timeout, so each star runs in its own background thread
and is abandoned (logged as a timeout, retryable) if it takes longer than --timeout seconds.
A few stars download at once (--workers).

Usage:
    python score_candidates.py --limit 20        # pilot: check timing first
    python score_candidates.py                   # everything (resumes automatically)
    python score_candidates.py --retry-failed    # give failed KOIs another try
"""
import argparse
import queue
import threading
import time
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd  # noqa: E402

from build_features import append_rows, names_in  # noqa: E402
from build_koi_list import CANDIDATES_PATH, DATA_DIR  # noqa: E402
from inference import analyze, has_local_data, training_kois  # noqa: E402
from train_classifier import FULL_FEATURES, SEED  # noqa: E402

PREDICTIONS_PATH = DATA_DIR / "candidate_predictions.csv"
FAILED_PATH = DATA_DIR / "candidate_failed.csv"


def score_star(kepid: int, names: list[str]) -> tuple[list, list]:
    """(prediction rows, failure rows) for one star's candidate KOIs."""
    star_split = training_kois()[1].get(kepid)  # train/test if another KOI on this star was used
    rows, failures = [], []
    for name in names:
        try:
            r = analyze(name)
        except ValueError as e:  # not enough usable data around this KOI; others may be fine
            failures.append({"kepid": kepid, "kepoi_name": name, "error": f"ValueError: {e}"})
            continue
        rows.append({
            "kepid": kepid, "kepoi_name": name, "star_in_training_data": star_split or "",
            "prob_classic": r.prob_classic, "prob_cnn": r.prob_cnn, "prob_ensemble": r.prob_ensemble,
            **{f: r.features[f] for f in FULL_FEATURES},
            "warnings": " | ".join(r.warnings),
        })
    return rows, failures


def worker(kepid: int, names: list[str], results: queue.Queue) -> None:
    try:
        results.put((kepid, *score_star(kepid, names)))
    except Exception as e:  # noqa: BLE001 -- whole star failed (usually the download)
        failures = [{"kepid": kepid, "kepoi_name": n, "error": f"{type(e).__name__}: {e}"} for n in names]
        results.put((kepid, [], failures))


def summarize() -> None:
    if PREDICTIONS_PATH.exists():
        p = pd.read_csv(PREDICTIONS_PATH)
        print(f"\n{len(p)} candidates scored. Ensemble says planet for "
              f"{(p['prob_ensemble'] >= 0.5).sum()} ({(p['prob_ensemble'] >= 0.5).mean():.0%}); "
              f"both models agree on {((p['prob_classic'] >= 0.5) == (p['prob_cnn'] >= 0.5)).mean():.0%}.")
    if FAILED_PATH.exists():
        f = pd.read_csv(FAILED_PATH)
        print(f"{len(f)} failed:")
        print(f["error"].str.split(":").str[0].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every CANDIDATE KOI with both models.")
    parser.add_argument("--limit", type=int, default=None, help="Only process this many stars")
    parser.add_argument("--workers", type=int, default=4, help="Stars downloading at once")
    parser.add_argument("--timeout", type=float, default=300, help="Give up on a star after this many seconds")
    parser.add_argument("--retry-failed", action="store_true", help="Retry KOIs in candidate_failed.csv")
    args = parser.parse_args()

    candidates = pd.read_csv(CANDIDATES_PATH)
    if args.retry_failed and FAILED_PATH.exists():
        FAILED_PATH.unlink()
    skip = names_in(PREDICTIONS_PATH) | names_in(FAILED_PATH)
    todo = candidates[~candidates["kepoi_name"].isin(skip)]
    stars = todo.groupby("kepid")["kepoi_name"].apply(list).sample(frac=1, random_state=SEED)
    if args.limit:
        stars = stars.iloc[: args.limit]
    local = sum(has_local_data(k) for k in stars.index)
    print(f"{len(skip)} KOIs already done or failed; processing {len(stars)} star(s) "
          f"({sum(map(len, stars))} KOIs, {local} already downloaded) with {args.workers} workers", flush=True)

    pending = list(stars.items())
    active = {}  # kepid -> start time
    results = queue.Queue()
    done, start = 0, time.perf_counter()
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                kepid, names = pending.pop(0)
                active[kepid] = time.perf_counter()
                # daemon: a download that never returns can't keep the script from exiting
                threading.Thread(target=worker, args=(kepid, names, results), daemon=True).start()

            try:
                kepid, rows, failures = results.get(timeout=5)
            except queue.Empty:
                kepid = None
            if kepid is not None and kepid in active:  # (an abandoned star finishing late is ignored)
                seconds = time.perf_counter() - active.pop(kepid)
                status = f"{len(rows)} ok" + (f", {len(failures)} failed ({failures[0]['error'][:80]})"
                                              if failures else "")
            else:
                stalled = [k for k, t0 in active.items() if time.perf_counter() - t0 > args.timeout]
                if not stalled:
                    continue
                kepid = stalled[0]
                seconds = time.perf_counter() - active.pop(kepid)
                rows, failures = [], [{"kepid": kepid, "kepoi_name": n,
                                       "error": f"Timeout: abandoned after {args.timeout:.0f} s"}
                                      for n in stars[kepid]]
                status = "TIMED OUT (abandoned; retry later with --retry-failed)"

            append_rows(PREDICTIONS_PATH, rows)
            append_rows(FAILED_PATH, failures)
            done += 1
            elapsed = time.perf_counter() - start
            eta_h = elapsed / done * (len(stars) - done) / 3600
            print(f"[{done}/{len(stars)}] KIC {kepid}: {status} ({seconds:.0f} s; "
                  f"{elapsed / done:.1f} s/star overall; ~{eta_h:.1f} h left)", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Everything finished so far is saved; re-run to resume.")

    summarize()


if __name__ == "__main__":
    main()
