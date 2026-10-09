"""Milestone 5: train the classic (feature-based) classifier and evaluate it honestly.

  1. Split stars 80/20 into train and test. The split is by STAR, not by KOI, so two
     signals from the same star never land on opposite sides (that would leak).
     The split is saved to data/split.csv so the CNN in milestone 6 uses the exact
     same test stars and the comparison is fair.
  2. Train two gradient-boosted tree models:
       - baseline: only the basic transit numbers (period, duration, depth)
       - full:     every vetting feature from milestone 4
     The gap between them is what the vetting features are worth.
  3. Report 5-fold cross-validation on the training set, then a single final score on
     the untouched test set, plus a confusion matrix, ROC curves, feature importance,
     and the model's most confident mistakes.

Usage:
    python train_classifier.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features.csv"
SPLIT_PATH = DATA_DIR / "split.csv"
PREDICTIONS_PATH = DATA_DIR / "test_predictions_classic.csv"
MODEL_DIR = Path("models")
PLOT_PATH = Path("plots") / "classic_results.png"
SEED = 42

BASELINE_FEATURES = ["period_days", "duration_hours", "depth_ppm"]
FULL_FEATURES = [
    "period_days", "duration_hours", "duration_frac", "depth_ppm", "depth_snr",
    "odd_even_sigma", "odd_even_rel", "secondary_snr", "secondary_ratio",
    "shape_ratio", "harmonic_dll", "max_single_frac", "n_transits", "noise_ppm",
]

# Plot colors: dataviz reference palette slots 1-2 + neutral ink
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#52514e", "#b5b3ad", "#e4e3df", "#fcfcfb"


def make_model() -> HistGradientBoostingClassifier:
    # Small, regularized trees: ~800 training rows is not much data. NaNs are handled natively.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
    )


def split_by_star(df: pd.DataFrame) -> pd.Series:
    """Return a 'train'/'test' label per row, reusing data/split.csv if it already exists."""
    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH).set_index("kepid")["split"]
        if set(df["kepid"]) <= set(saved.index):
            print(f"Reusing existing split: {SPLIT_PATH}")
            return df["kepid"].map(saved)
        print("Existing split doesn't cover every star; making a new one.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, _ = next(splitter.split(df, df["label"], groups=df["kepid"]))
    split = pd.Series("test", index=df.index)
    split.iloc[train_idx] = "train"
    df[["kepid"]].assign(split=split).drop_duplicates("kepid").to_csv(SPLIT_PATH, index=False)
    print(f"Saved new star-level split: {SPLIT_PATH}")
    return split


def scores(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
    }


def print_scores(title: str, rows: dict) -> None:
    print(f"\n{title}")
    print(f"  {'model':<10}{'accuracy':>10}{'precision':>11}{'recall':>9}{'F1':>8}{'ROC AUC':>10}")
    for name, s in rows.items():
        print(f"  {name:<10}{s['accuracy']:>10.3f}{s['precision']:>11.3f}{s['recall']:>9.3f}"
              f"{s['f1']:>8.3f}{s['roc_auc']:>10.3f}")


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def plot_results(y_test, probs: dict, cm: np.ndarray, importance: pd.Series) -> None:
    fig, (ax_cm, ax_roc, ax_imp) = plt.subplots(
        1, 3, figsize=(17, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [0.8, 1, 1.2]})

    # 1. Confusion matrix (full model, test set)
    style(ax_cm, "Full model: test-set confusion matrix")
    ax_cm.imshow(cm, cmap="Blues", vmin=0)
    labels = ["False positive", "Planet"]
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                       color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax_cm.set_xticks([0, 1], labels)
    ax_cm.set_yticks([0, 1], labels)
    ax_cm.set_xlabel("Predicted", color=INK, fontsize=9)
    ax_cm.set_ylabel("Actual", color=INK, fontsize=9)

    # 2. ROC curves, both models
    style(ax_roc, "ROC curve (test set)")
    ax_roc.grid(True, color=GRID, linewidth=0.6)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random guess")
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        color = ORANGE if name == "baseline" else BLUE
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{name} (AUC {roc_auc_score(y_test, prob):.3f})")
    ax_roc.set_xlabel("False positive rate (fakes called planets)", color=INK, fontsize=9)
    ax_roc.set_ylabel("True positive rate (planets caught)", color=INK, fontsize=9)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    # 3. Permutation importance (how much test AUC drops when a feature is shuffled)
    style(ax_imp, "Feature importance (drop in AUC when shuffled)")
    ax_imp.grid(True, axis="x", color=GRID, linewidth=0.6)
    imp = importance.sort_values()
    ax_imp.barh(imp.index, imp.values, color=BLUE, height=0.7)
    ax_imp.set_axisbelow(True)

    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nSaved {PLOT_PATH}")


def main() -> None:
    df = pd.read_csv(FEATURES_PATH).replace([np.inf, -np.inf], np.nan)
    df["split"] = split_by_star(df)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    print(f"Train: {len(train)} KOIs on {train['kepid'].nunique()} stars "
          f"({train['label'].mean():.0%} planets)")
    print(f"Test:  {len(test)} KOIs on {test['kepid'].nunique()} stars "
          f"({test['label'].mean():.0%} planets)")

    # Cross-validation on the training set only (folds also grouped by star)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    for name, cols in (("baseline", BASELINE_FEATURES), ("full", FULL_FEATURES)):
        prob = cross_val_predict(make_model(), train[cols], train["label"], groups=train["kepid"],
                                 cv=cv, method="predict_proba")[:, 1]
        cv_scores[name] = scores(train["label"], prob)
    print_scores("5-fold cross-validation (training set)", cv_scores)

    # Final fit on all training data, one evaluation on the untouched test set
    models, test_probs, test_scores = {}, {}, {}
    for name, cols in (("baseline", BASELINE_FEATURES), ("full", FULL_FEATURES)):
        models[name] = make_model().fit(train[cols], train["label"])
        test_probs[name] = models[name].predict_proba(test[cols])[:, 1]
        test_scores[name] = scores(test["label"], test_probs[name])
    print_scores("Test set (never seen during training)", test_scores)

    full_prob = test_probs["full"]
    cm = confusion_matrix(test["label"], (full_prob >= 0.5).astype(int))
    print(f"\nConfusion matrix (full model): {cm[1, 1]} planets caught, {cm[1, 0]} planets missed, "
          f"{cm[0, 0]} fakes rejected, {cm[0, 1]} fakes let through")

    result = permutation_importance(models["full"], test[FULL_FEATURES], test["label"],
                                    scoring="roc_auc", n_repeats=20, random_state=SEED)
    importance = pd.Series(result.importances_mean, index=FULL_FEATURES)
    print("\nTop features (drop in test AUC when shuffled):")
    for feat, val in importance.sort_values(ascending=False).head(6).items():
        print(f"  {feat:<16} {val:.3f}")

    # Most confident mistakes: worth opening with extract_features.py to see why
    out = test[["kepid", "kepoi_name", "label", "koi_disposition"]].assign(prob_planet=full_prob)
    wrong = out[(out["prob_planet"] >= 0.5) != (out["label"] == 1)]
    wrong = wrong.assign(confidence=(wrong["prob_planet"] - 0.5).abs()).sort_values("confidence", ascending=False)
    print("\nMost confident mistakes:")
    for r in wrong.head(6).itertuples():
        print(f"  {r.kepoi_name}  actually {r.koi_disposition:<15} model said planet with p={r.prob_planet:.2f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": models["full"], "features": FULL_FEATURES}, MODEL_DIR / "classic_full.joblib")
    out.to_csv(PREDICTIONS_PATH, index=False)
    (MODEL_DIR / "classic_metrics.json").write_text(json.dumps(
        {"cv": cv_scores, "test": test_scores}, indent=2))
    print(f"\nSaved {MODEL_DIR / 'classic_full.joblib'}, {PREDICTIONS_PATH}, {MODEL_DIR / 'classic_metrics.json'}")

    plot_results(test["label"], test_probs, cm, importance)


if __name__ == "__main__":
    main()
