"""Milestone 8: what the models say about the 1,873 held-out CANDIDATE KOIs, and how far to trust it.

Main caveat, measured here: confirmed planets in the training set are mostly strong signals
(confirming a planet needs follow-up, which favors them), so the training set has almost no
faint real planets. The models learned "faint signal -> false positive", and most candidates
are faint. Verdicts are only meaningful where training had both classes, roughly SNR > 15.

Prints:
  1. how many candidates each model calls a planet, and how often the two models agree
  2. planet share by signal-to-noise: training labels vs. our candidate verdicts vs. NASA's
     Robovetter score (NASA's automated vetting; every candidate already passed it)
  3. agreement with the Robovetter where our verdicts are trustworthy
  4. the top-flagged candidates
Saves data/candidates_flagged.csv (all candidates the ensemble calls planets, SNR > 15)
and plots/candidates_snr.png.

Usage:
    python analyze_candidates.py
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from build_features import FEATURES_PATH  # noqa: E402
from build_koi_list import CANDIDATES_PATH, DATA_DIR  # noqa: E402
from robovetter import load_flags  # noqa: E402
from score_candidates import PREDICTIONS_PATH  # noqa: E402
from train_classifier import BLUE, INK, ORANGE, SURFACE, style  # noqa: E402
from train_cnn import CNN_PREDICTIONS  # noqa: E402

FLAGGED_PATH = DATA_DIR / "candidates_flagged.csv"
PLOT_PATH = DATA_DIR.parent / "plots" / "candidates_snr.png"
TRUSTED_SNR = 15   # below this, training had almost no real planets
SNR_BINS = [0, 10, 15, 20, 30, 50, np.inf]
SNR_LABELS = ["≤10", "10–15", "15–20", "20–30", "30–50", ">50"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = pd.read_csv(PREDICTIONS_PATH).merge(load_flags()[["kepoi_name", "koi_score"]], on="kepoi_name",
                                               how="left")
    cand["says_planet"] = cand["prob_ensemble"] >= 0.5
    cand["snr_bin"] = pd.cut(cand["depth_snr"], SNR_BINS, labels=SNR_LABELS)
    train = pd.read_csv(FEATURES_PATH)
    train["snr_bin"] = pd.cut(train["depth_snr"], SNR_BINS, labels=SNR_LABELS)
    return cand, train


def overview(cand: pd.DataFrame) -> None:
    total = len(pd.read_csv(CANDIDATES_PATH))
    test = pd.read_csv(CNN_PREDICTIONS)
    agree = ((cand["prob_classic"] >= 0.5) == (cand["prob_cnn"] >= 0.5)).mean()
    agree_test = ((test["prob_classic"] >= 0.5) == (test["prob_cnn"] >= 0.5)).mean()
    print(f"Scored {len(cand)} of {total} candidates.")
    for model in ("classic", "cnn", "ensemble"):
        print(f"  {model:<9} calls {(cand[f'prob_{model}'] >= 0.5).sum():>4} planets "
              f"({(cand[f'prob_{model}'] >= 0.5).mean():.0%})")
    print(f"  classic and CNN agree on {agree:.0%} of candidates vs {agree_test:.0%} of the test set "
          "(more disagreement = inputs unlike the training data)")


def by_snr(cand: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "training_kois": train.groupby("snr_bin", observed=False).size(),
        "training_planet_share": train.groupby("snr_bin", observed=False)["label"].mean(),
        "candidates": cand.groupby("snr_bin", observed=False).size(),
        "candidates_called_planet": cand.groupby("snr_bin", observed=False)["says_planet"].mean(),
        "robovetter_mean_score": cand.groupby("snr_bin", observed=False)["koi_score"].mean(),
    })
    print(f"\nBy signal-to-noise (median SNR: candidates {cand['depth_snr'].median():.1f}, "
          f"training planets {train.loc[train['label'] == 1, 'depth_snr'].median():.1f}):")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))
    return table


def robovetter_agreement(cand: pd.DataFrame) -> None:
    scored = cand.dropna(subset=["koi_score"])
    trusted = scored[scored["depth_snr"] > TRUSTED_SNR]
    print(f"\nRobovetter score available for {len(scored)} of {len(cand)} candidates.")
    for label, df in (("all", scored), (f"SNR > {TRUSTED_SNR}", trusted)):
        rho = spearmanr(df["prob_ensemble"], df["koi_score"]).statistic
        agree = (df["says_planet"] == (df["koi_score"] >= 0.5)).mean()
        print(f"  {label:<9} n={len(df):>4}  Spearman {rho:.2f}  same verdict (score >= 0.5) {agree:.0%}")
    print(pd.crosstab(trusted["says_planet"].map({True: "we say planet", False: "we say FP"}),
                      (trusted["koi_score"] >= 0.5).map({True: "Robovetter >= 0.5", False: "Robovetter < 0.5"})
                      ).to_string())


def flagged(cand: pd.DataFrame) -> None:
    keep = cand[cand["says_planet"] & (cand["depth_snr"] > TRUSTED_SNR)]
    cols = ["kepoi_name", "kepid", "prob_ensemble", "prob_classic", "prob_cnn", "koi_score", "period_days",
            "depth_ppm", "depth_snr", "duration_hours", "star_in_training_data", "warnings"]
    keep = keep.sort_values("prob_ensemble", ascending=False)[cols]
    keep.to_csv(FLAGGED_PATH, index=False)
    both = keep[(keep["prob_classic"] >= 0.9) & (keep["prob_cnn"] >= 0.9)]
    print(f"\n{len(keep)} candidates with SNR > {TRUSTED_SNR} that the ensemble calls planets "
          f"({len(both)} with both models >= 0.9); saved {FLAGGED_PATH}")
    print(keep.head(15).drop(columns=["kepid", "warnings"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))


def plot(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    x = np.arange(len(table))
    width = 0.36
    for offset, col, color, label in ((-width / 2, "training_planet_share", BLUE,
                                       "Training KOIs that were real planets"),
                                      (width / 2, "candidates_called_planet", ORANGE,
                                       "Candidates our ensemble calls planets")):
        bars = ax.bar(x + offset, table[col], width * 0.94, color=color, label=label)
        for bar, value in zip(bars, table[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.0%}", ha="center",
                    va="bottom", color=INK, fontsize=8)
    ax.set_xticks(x, [f"{lab}\n{n} cand." for lab, n in zip(table.index, table["candidates"])])
    ax.set_xlabel("Signal-to-noise of the transit (depth_snr)", color=INK, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.axvspan(-0.5, 1.5, color=INK, alpha=0.06, lw=0)
    ax.text(0.5, 0.93, "too few faint planets\nin training", ha="center", va="top", color=INK, fontsize=8)
    style(ax, "The models learned that faint signals are false positives; most candidates are faint")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved {PLOT_PATH}")


def main() -> None:
    cand, train = load()
    overview(cand)
    table = by_snr(cand, train)
    robovetter_agreement(cand)
    flagged(cand)
    plot(table)


if __name__ == "__main__":
    main()
