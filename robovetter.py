"""Milestone 8: compare our models with NASA's own automated vetting (the Kepler Robovetter).

Fetches these columns of the cumulative KOI table into a separate file, so the training
catalog (koi_cumulative.csv) never changes:
  koi_score       Robovetter score, 0-1: its confidence that the KOI is a planet candidate
  koi_fpflag_nt   "not transit-like": instrumental glitches, stellar variability
  koi_fpflag_ss   "stellar eclipse": secondary eclipse, odd/even difference, V shape
  koi_fpflag_co   "centroid offset": the dip comes from a nearby star, not the target
  koi_fpflag_ec   "ephemeris match": light leaking from another known variable star

Question tested here: are our models' confident mistakes the blended-binary kind? If so,
false positives we call planets should carry the centroid-offset flag far more often than
the false positives we catch. Uses the classic model's predictions for all 994 labeled KOIs:
out-of-fold for training KOIs (same 5-fold star-grouped CV as train_classifier.py, so each
prediction comes from a model that never saw that KOI) and the saved test predictions.

Caveat: NASA's FALSE POSITIVE dispositions are largely *based on* these flags, so they say
how NASA caught each impostor; they are not an independent judge of our models.

Usage:
    python robovetter.py              # fetch flags if needed, run the mistake analysis
    python robovetter.py --refresh    # re-download the flags
"""
import argparse
import io
import urllib.parse
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402

from build_koi_list import DATA_DIR, TAP_URL  # noqa: E402
from inference import labeled_features  # noqa: E402
from train_classifier import (FULL_FEATURES, INK, PREDICTIONS_PATH, SEED, SURFACE, make_model,  # noqa: E402
                              style)

FLAGS_PATH = DATA_DIR / "koi_vetting_flags.csv"
PLOT_PATH = DATA_DIR.parent / "plots" / "mistakes_vs_nasa_flags.png"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]  # dataviz blue ramp, light -> dark = less -> more confident miss
FLAGS = {"koi_fpflag_nt": "not transit-like", "koi_fpflag_ss": "stellar eclipse",
         "koi_fpflag_co": "centroid offset", "koi_fpflag_ec": "ephemeris match"}
COLUMNS = ["kepoi_name", "koi_score", *FLAGS]


def load_flags(refresh: bool = False) -> pd.DataFrame:
    if FLAGS_PATH.exists() and not refresh:
        return pd.read_csv(FLAGS_PATH)
    query = f"select {','.join(COLUMNS)} from cumulative"
    url = f"{TAP_URL}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"
    print("Downloading Robovetter scores and flags from NASA Exoplanet Archive...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        flags = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
    flags.to_csv(FLAGS_PATH, index=False)
    print(f"Saved {FLAGS_PATH} ({len(flags)} KOIs)")
    return flags


def classic_predictions() -> pd.DataFrame:
    """P(planet) from the classic model for all labeled KOIs, each from a model that never saw it."""
    df = labeled_features()
    train = df[df["split"] == "train"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_model(), train[FULL_FEATURES], train["label"], groups=train["kepid"],
                            cv=cv, method="predict_proba")[:, 1]
    test = pd.read_csv(PREDICTIONS_PATH)[["kepoi_name", "prob_planet"]]
    return pd.concat([
        train[["kepoi_name", "label", "split"]].assign(prob=oof),
        df[df["split"] == "test"][["kepoi_name", "label", "split"]].merge(
            test.rename(columns={"prob_planet": "prob"}), on="kepoi_name"),
    ])


def mistake_analysis(flags: pd.DataFrame) -> None:
    preds = classic_predictions().merge(flags, on="kepoi_name", how="left")
    fps = preds[preds["label"] == 0].copy()
    fps["group"] = pd.cut(fps["prob"], [-0.01, 0.5, 0.9, 1.01],
                          labels=["caught (p < 0.5)", "missed (0.5-0.9)", "missed, confident (p > 0.9)"])

    print(f"\nFalse positives: {len(fps)} ({(fps['split'] == 'test').sum()} test, "
          f"{(fps['split'] == 'train').sum()} training out-of-fold)")
    table = fps.groupby("group", observed=True).agg(
        n=("kepoi_name", "size"), **{name: (flag, "mean") for flag, name in FLAGS.items()},
        robovetter_score=("koi_score", "mean"))
    print("\nShare of each group carrying each NASA flag (a KOI can have several):")
    print(table.to_string(float_format=lambda x: f"{x:.0%}" if x <= 1 else f"{x:.0f}"))

    missed = fps["prob"] >= 0.5
    co = fps["koi_fpflag_co"] == 1
    ss = fps["koi_fpflag_ss"] == 1
    for name, flag in (("centroid offset", co), ("stellar eclipse", ss)):
        odds, p = fisher_exact([[(missed & flag).sum(), (missed & ~flag).sum()],
                                [(~missed & flag).sum(), (~missed & ~flag).sum()]])
        print(f"\n{name}: {flag[missed].mean():.0%} of missed vs {flag[~missed].mean():.0%} of caught "
              f"(odds ratio {odds:.1f}, Fisher exact p = {p:.2g})")

    print("\nMost confident misses:")
    worst = fps[missed].sort_values("prob", ascending=False).head(12)
    for r in worst.itertuples():
        raised = [name for flag, name in FLAGS.items() if getattr(r, flag) == 1] or ["none"]
        print(f"  {r.kepoi_name} ({r.split:<5}) p={r.prob:.3f}  flags: {', '.join(raised)}")

    plot_flags(table)


def plot_flags(table: pd.DataFrame) -> None:
    """One small panel per NASA flag: share of each false-positive group carrying it."""
    fig, axes = plt.subplots(1, len(FLAGS), figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    groups = list(table.index)
    for ax, name in zip(axes, FLAGS.values()):
        shares = table[name].to_numpy()
        bars = ax.bar(range(len(groups)), shares, width=0.6, color=RAMP)
        for bar, share in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, share + 0.02, f"{share:.0%}",
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        style(ax, name)
    axes[0].set_ylabel("Share of false positives", color=INK, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in RAMP]
    labels = [f"{g} (n={n})" for g, n in zip(groups, table["n"])]
    fig.suptitle("NASA's vetting flags on the false positives our classic model caught vs. missed",
                 x=0.01, y=0.98, ha="left", color=INK, fontsize=11)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.91), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved {PLOT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare with NASA's Robovetter flags.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the flags")
    args = parser.parse_args()
    mistake_analysis(load_flags(args.refresh))


if __name__ == "__main__":
    main()
