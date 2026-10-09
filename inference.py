"""Milestone 7: score any catalog KOI with both models (the dashboard's engine).

Runs exactly the path the training data went through, for one KOI:
  light curve (local cache, else MAST) -> clean once per star with every KOI masked
  -> remove other KOIs' transits -> vetting features + CNN views -> classic, CNN, ensemble

It also says which part of the data the KOI belongs to, because a verdict on a KOI the
model trained on proves little:
  train      the models learned from it
  test       held out from training; the reported test scores come from these 196
  unseen     a CONFIRMED / FALSE POSITIVE KOI that wasn't in the 1,000 we sampled,
             so it's a fresh out-of-sample check
  candidate  NASA hasn't decided yet; there is no true label

Usage:
    python inference.py K00072.01         # Kepler-10b
    python inference.py "KIC 11904151"    # every KOI on that star
    python inference.py "Kepler-10"       # same, by planet-system name
    python inference.py --parity          # re-score the test set, compare with saved results
"""
import argparse
import contextlib
import io
import re
import time
import warnings
from dataclasses import dataclass, field
from functools import cache, lru_cache

warnings.filterwarnings("ignore", message=".*oktopus.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from astroquery.mast import Observations  # noqa: E402

from build_features import CACHE_DIR, FEATURES_PATH, MIN_TRANSITS  # noqa: E402
from build_koi_list import MAX_DURATION_HOURS  # noqa: E402
from build_views import load_cached, make_views  # noqa: E402
from extract_features import extract_features, koi_data, load_catalog, prepare_star, star_ephemerides  # noqa: E402
from fetch_lightcurve import fetch  # noqa: E402
from train_classifier import FULL_FEATURES, MODEL_DIR, SPLIT_PATH  # noqa: E402
from train_cnn import CNN_PREDICTIONS, TransitCNN, as_tensor, predict  # noqa: E402

# astroquery reads its 600 s search timeout once at import, so set it on the live connection.
# (File downloads have no timeout at all in astroquery; nothing we can do about those here.)
MAST_TIMEOUT = 90
Observations._portal_api_connection.TIMEOUT = MAST_TIMEOUT


@dataclass
class KoiResult:
    kepoi_name: str
    kepid: int
    kepler_name: str | None
    disposition: str            # CONFIRMED / FALSE POSITIVE / CANDIDATE
    dataset: str                # train / test / unseen / candidate
    period: float               # days
    t0: float                   # mid-transit time, BKJD
    duration: float             # days
    ephemerides: dict           # every KOI on this star: name -> (period, t0, duration days)
    star_t: np.ndarray          # whole star, cleaned (all KOIs still in)
    star_y: np.ndarray
    t: np.ndarray               # this KOI only (other KOIs' transits removed)
    y: np.ndarray
    features: dict
    global_view: np.ndarray
    local_view: np.ndarray
    prob_classic: float
    prob_cnn: float
    warnings: list[str] = field(default_factory=list)

    @property
    def prob_ensemble(self) -> float:
        return (self.prob_classic + self.prob_cnn) / 2

    @property
    def label(self) -> int | None:
        return {"CONFIRMED": 1, "FALSE POSITIVE": 0}.get(self.disposition)


# ---------- Loaded once per process ----------

@cache
def catalog() -> pd.DataFrame:
    return load_catalog()


@cache
def models():
    classic = joblib.load(MODEL_DIR / "classic_full.joblib")
    cnn = TransitCNN()
    cnn.load_state_dict(torch.load(MODEL_DIR / "cnn.pt", weights_only=True))
    cnn.eval()
    return classic, cnn


@cache
def training_kois() -> tuple[set, pd.Series]:
    """(KOIs the models were trained/tested on, kepid -> train/test)."""
    names = set(pd.read_csv(FEATURES_PATH, usecols=["kepoi_name"])["kepoi_name"])
    split = pd.read_csv(SPLIT_PATH).set_index("kepid")["split"]
    return names, split


def dataset_of(name: str, kepid: int, disposition: str) -> str:
    names, split = training_kois()
    if name in names:
        return split[kepid]
    return "candidate" if disposition == "CANDIDATE" else "unseen"


# ---------- Lookup ----------

def find_kois(query: str) -> list[str]:
    """Catalog KOI names matching a KOI (K00072.01, KOI-72.01, 72.01), a star (KIC 11904151,
    K00072) or a Kepler name (Kepler-10 b, Kepler-10). Empty list if nothing matches."""
    cat, q = catalog(), query.strip()

    m = re.fullmatch(r"(?:KOI|K)?[-\s]*0*(\d+)\.(\d{1,2})", q, re.IGNORECASE)
    if m:
        name = f"K{int(m[1]):05d}.{int(m[2]):02d}"
        return [name] if name in set(cat["kepoi_name"]) else []

    m = re.fullmatch(r"(?:KOI|K)[-\s]*0*(\d+)", q, re.IGNORECASE)
    if m:
        rows = cat[cat["kepoi_name"].str.startswith(f"K{int(m[1]):05d}.")]
    elif m := re.fullmatch(r"(?:KIC)?[-\s]*(\d+)", q, re.IGNORECASE):
        rows = cat[cat["kepid"] == int(m[1])]
    else:
        names = cat["kepler_name"].fillna("").str.lower()
        q = q.lower()
        rows = cat[(names == q) | names.str.startswith(q + " ")]
    return sorted(rows["kepoi_name"])


# ---------- Pipeline ----------

def load_light_curve(kepid: int):
    """Local cache first (instant); otherwise download from MAST into the same cache."""
    try:
        return load_cached(kepid)
    except FileNotFoundError:
        with contextlib.redirect_stdout(io.StringIO()):  # keep fetch()'s progress chatter quiet
            return fetch(f"KIC {kepid}", None, download_dir=str(CACHE_DIR))


@lru_cache(maxsize=8)
def cleaned_star(kepid: int):
    """(ephemerides, t, y, dy) for a star, cleaned once with every KOI masked from the trend."""
    ephemerides = star_ephemerides(kepid, catalog())
    return (ephemerides, *prepare_star(load_light_curve(kepid), ephemerides))


def analyze(name: str) -> KoiResult:
    row = catalog().set_index("kepoi_name").loc[name]
    kepid = int(row["kepid"])
    ephemerides, star_t, star_y, star_dy = cleaned_star(kepid)
    if name not in ephemerides:
        raise ValueError(f"{name} has no usable period / t0 / duration in the catalog")
    period, t0, duration = ephemerides[name]

    t, y, dy = koi_data(star_t, star_y, star_dy, ephemerides, name)
    features = extract_features(t, y, dy, period, t0, duration)
    global_view, local_view = make_views(t, y, period, t0, duration)

    classic, cnn = models()
    x = pd.DataFrame([features])[classic["features"]].replace([np.inf, -np.inf], np.nan)
    prob_classic = float(classic["model"].predict_proba(x)[0, 1])
    prob_cnn = float(predict(cnn, as_tensor(global_view[None]), as_tensor(local_view[None]))[0])

    notes = []
    if duration * 24 > MAX_DURATION_HOURS:
        notes.append(f"Transit lasts {duration * 24:.1f} h; training only used transits up to "
                     f"{MAX_DURATION_HOURS} h, and the ~2-day flattening can erase part of long ones.")
    if features["n_transits"] < MIN_TRANSITS:
        notes.append(f"Only {features['n_transits']} transit(s) in the data; every training KOI "
                     f"had at least {MIN_TRANSITS}, so treat this verdict as unreliable.")

    kepler_name = row["kepler_name"] if isinstance(row["kepler_name"], str) else None
    return KoiResult(
        kepoi_name=name, kepid=kepid, kepler_name=kepler_name,
        disposition=row["koi_disposition"],
        dataset=dataset_of(name, kepid, row["koi_disposition"]),
        period=period, t0=t0, duration=duration, ephemerides=ephemerides,
        star_t=star_t, star_y=star_y, t=t, y=y,
        features=features, global_view=global_view, local_view=local_view,
        prob_classic=prob_classic, prob_cnn=prob_cnn, warnings=notes,
    )


# ---------- CLI ----------

def print_result(r: KoiResult, seconds: float) -> None:
    title = f"{r.kepoi_name}" + (f" ({r.kepler_name})" if r.kepler_name else "")
    print(f"\n{title} on KIC {r.kepid}: {r.disposition}, dataset = {r.dataset}  [{seconds:.1f} s]")
    print(f"  period {r.period:.5f} d, duration {r.duration * 24:.2f} h, "
          f"{r.features['n_transits']} transits, depth {r.features['depth_ppm']:.0f} ppm")
    print(f"  P(planet): classic {r.prob_classic:.3f} | CNN {r.prob_cnn:.3f} | ensemble {r.prob_ensemble:.3f}")
    for note in r.warnings:
        print(f"  WARNING: {note}")


def parity_check() -> None:
    """Re-score every test KOI through analyze() and compare with what training saved."""
    saved = pd.read_csv(CNN_PREDICTIONS).sort_values("kepid")  # sorted so each star is cleaned once
    saved_features = pd.read_csv(FEATURES_PATH).set_index("kepoi_name")
    rows, start = [], time.perf_counter()
    for i, name in enumerate(saved["kepoi_name"], 1):
        r = analyze(name)
        new = pd.Series(r.features)[FULL_FEATURES].astype(float).to_numpy()
        old = saved_features.loc[name, FULL_FEATURES].astype(float).to_numpy()
        close = np.isclose(new, old, rtol=1e-6, equal_nan=True)
        rows.append({"kepoi_name": name, "new_classic": r.prob_classic, "new_cnn": r.prob_cnn,
                     "features_differ": ", ".join(np.array(FULL_FEATURES)[~close])})
        if i % 20 == 0 or i == len(saved):
            print(f"[{i}/{len(saved)}] {time.perf_counter() - start:.0f} s")

    both = saved.merge(pd.DataFrame(rows), on="kepoi_name")
    d_classic = (both["new_classic"] - both["prob_classic"]).abs()
    d_cnn = (both["new_cnn"] - both["prob_cnn"]).abs()
    new_ensemble = (both["new_classic"] + both["new_cnn"]) / 2
    flipped = ((new_ensemble >= 0.5) != (both["prob_ensemble"] >= 0.5)).sum()
    feature_mismatch = both[both["features_differ"] != ""]

    print(f"\nParity on {len(both)} test KOIs:")
    print(f"  classic  max |diff| {d_classic.max():.2e}")
    print(f"  CNN      max |diff| {d_cnn.max():.2e}")
    print(f"  ensemble verdicts that flipped: {flipped}")
    print(f"  KOIs whose features differ (rtol 1e-6): {len(feature_mismatch)}")
    for r in feature_mismatch.head(10).itertuples():
        print(f"    {r.kepoi_name}: {r.features_differ}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a catalog KOI with the classic + CNN models.")
    parser.add_argument("query", nargs="?", default="K00072.01", help="KOI, KIC or Kepler name")
    parser.add_argument("--parity", action="store_true", help="Re-score the test set and compare")
    args = parser.parse_args()

    if args.parity:
        parity_check()
        return

    names = find_kois(args.query)
    if not names:
        raise SystemExit(f"No KOI in the catalog matches '{args.query}'.")
    print(f"'{args.query}' -> {', '.join(names)}")
    for name in names:
        start = time.perf_counter()
        r = analyze(name)
        print_result(r, time.perf_counter() - start)


if __name__ == "__main__":
    main()
