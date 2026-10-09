"""Milestone 7: precompute the dashboard's showcase KOIs so they load instantly.

For each showcase star, saves its *cleaned* light curve to data/demo/kic<KIC>.npz (~1 MB,
full float64 precision so results match the full pipeline exactly). Features, CNN views and
predictions are still computed live from these arrays, so they work without the 6 GB FITS
cache (e.g. on a hosted server). Also writes data/demo/showcase.csv for the app's sidebar.

After saving, every showcase KOI is re-scored from the saved file and compared with the
full pipeline; any difference stops the script.

Rebuild after changing the cleaning step or refreshing the KOI catalog.

Usage:
    python build_demo.py
"""
import time
import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from inference import DEMO_DIR, analyze, cleaned_star, demo_path, kepid_of  # noqa: E402

SHOWCASE_PATH = DEMO_DIR / "showcase.csv"
SHOWCASE = [  # (KOI, short title, why it's worth a look)
    ("K00072.01", "Kepler-10 b",
     "Kepler's first confirmed rocky planet, on a 20-hour orbit. In the training set."),
    ("K00072.02", "Kepler-10 c",
     "Same star, much wider orbit. Never trained on, though its star's light curve was."),
    ("K00289.01", "Kepler-511 c",
     "A test-set planet the models call correctly and confidently."),
    ("K01133.01", "Eclipsing binary",
     "Classic V-shaped eclipse, and its depth changes with Kepler's seasons."),
    ("K06662.01", "Glowing companion",
     "Test-set false positive with a clear dip half an orbit later: the companion is a star."),
    ("K01262.01", "Confident mistake",
     "Test-set false positive that every model calls a planet; nothing in the brightness data gives it away."),
    ("K02034.02", "Missed planet",
     "A real planet (Kepler-1065 c) that both models reject; its half-orbit window shows a dip."),
    ("K00403.01", "Models disagree",
     "The trees say planet, the CNN says false positive. The CNN is right."),
    ("K01439.01", "Very long transit",
     "A 24-hour transit, longer than anything in training, so the verdict comes with a warning."),
    ("K00753.01", "Undecided candidate",
     "NASA hasn't ruled on this one yet; there's no right answer to check against."),
]


def probabilities(name: str) -> tuple[float, float]:
    r = analyze(name)
    return r.prob_classic, r.prob_cnn


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    stars = sorted({kepid_of(name) for name, _, _ in SHOWCASE})

    # 1. Score everything through the full pipeline (no demo files, so nothing stale is reused)
    for kepid in stars:
        demo_path(kepid).unlink(missing_ok=True)
    cleaned_star.cache_clear()
    start = time.perf_counter()
    full = {name: probabilities(name) for name, _, _ in SHOWCASE}
    print(f"Scored {len(full)} showcase KOIs via the full pipeline in {time.perf_counter() - start:.0f} s")

    # 2. Save each star's cleaned light curve
    for kepid in stars:
        _, t, y, dy = cleaned_star(kepid)
        np.savez_compressed(demo_path(kepid), t=t, y=y, dy=dy)
    size = sum(demo_path(k).stat().st_size for k in stars) / 1e6
    print(f"Saved {len(stars)} stars to {DEMO_DIR} ({size:.1f} MB)")

    # 3. Re-score from the saved files and require identical results
    cleaned_star.cache_clear()
    rows = []
    for name, title, why in SHOWCASE:
        r = analyze(name)
        if (r.prob_classic, r.prob_cnn) != full[name]:
            raise SystemExit(f"{name}: saved light curve gives different probabilities "
                             f"({r.prob_classic}, {r.prob_cnn}) vs {full[name]}")
        rows.append({"kepoi_name": name, "title": title, "why": why, "dataset": r.dataset,
                     "disposition": r.disposition, "classic": round(r.prob_classic, 3),
                     "cnn": round(r.prob_cnn, 3), "ensemble": round(r.prob_ensemble, 3)})
    table = pd.DataFrame(rows)
    table[["kepoi_name", "title", "why"]].to_csv(SHOWCASE_PATH, index=False)
    print(f"Re-scored from saved files: identical. Wrote {SHOWCASE_PATH}\n")
    print(table.drop(columns="why").to_string(index=False))


if __name__ == "__main__":
    main()
