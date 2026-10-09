"""Milestone 6b: train a 1D convolutional neural network (CNN) on the global + local views.

Architecture (a lighter version of Google's AstroNet):

    global view (2001) -> 4 conv blocks ->|
                                          |-> concat -> dense 128 -> dropout -> P(planet)
    local view  (201)  -> 2 conv blocks ->|

Each conv block = two 1D convolutions + max-pooling. The network learns its own
"features" from the raw shapes instead of using the hand-made ones from milestone 4.

Training details:
  - Same star-level train/test split as milestone 5 (data/split.csv), so results compare fairly.
  - 15% of the training stars are held out as a validation set. After every epoch the model
    is scored on them, and the best-scoring version is kept (early stopping) so it doesn't
    just memorize the training set.
  - Augmentation: each training example is randomly flipped left-to-right, since a transit
    looks the same played backwards. This doubles the variety of examples for free.

Also scores an "ensemble" that averages the CNN's and the classic model's probabilities.

Usage:
    python train_cnn.py
    python train_cnn.py --epochs 120 --patience 20
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from train_classifier import (BLUE, GRID, INK, MODEL_DIR, MUTED, ORANGE, SPLIT_PATH, SURFACE,
                              print_scores, scores, style)
from train_classifier import PREDICTIONS_PATH as CLASSIC_PREDICTIONS

VIEWS_PATH = Path("data") / "views.npz"
CNN_PREDICTIONS = Path("data") / "test_predictions_cnn.csv"
PLOT_PATH = Path("plots") / "cnn_results.png"
AQUA = "#1baf7a"  # dataviz reference palette slot 3
SEED = 42
BATCH = 32


# ---------- Model ----------

def conv_block(c_in: int, c_out: int, kernel: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.ReLU(),
        nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2), nn.ReLU(),
        nn.MaxPool1d(pool, stride=2),
    )


class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_branch = nn.Sequential(
            conv_block(1, 8, 5, 5), conv_block(8, 16, 5, 5),
            conv_block(16, 32, 5, 5), conv_block(32, 64, 5, 5),
            nn.AdaptiveMaxPool1d(8),
        )
        self.local_branch = nn.Sequential(
            conv_block(1, 16, 5, 7), conv_block(16, 32, 5, 7),
            nn.AdaptiveMaxPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 8 + 32 * 8, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1),
        )

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(global_view).flatten(1)
        loc = self.local_branch(local_view).flatten(1)
        return self.head(torch.cat([g, loc], dim=1)).squeeze(1)  # logits


# ---------- Training ----------

def as_tensor(views: np.ndarray) -> torch.Tensor:
    return torch.tensor(views, dtype=torch.float32).unsqueeze(1)  # (N, 1 channel, length)


def predict(model: nn.Module, g: torch.Tensor, loc: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(g, loc)).numpy()


def train_model(train, val, epochs: int, patience: int):
    (g_tr, l_tr, y_tr), (g_va, l_va, y_va) = train, val
    model = TransitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc, best_state, best_epoch, wait, history = -1.0, None, 0, 0, []
    n = len(y_tr)
    print(f"\n{'epoch':>5} {'train loss':>11} {'val AUC':>9}")
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            g, loc, y = g_tr[idx], l_tr[idx], y_tr[idx]
            flip = (torch.rand(len(idx)) < 0.5)[:, None, None]  # time-reversal augmentation
            g = torch.where(flip, g.flip(-1), g)
            loc = torch.where(flip, loc.flip(-1), loc)

            optimizer.zero_grad()
            loss = loss_fn(model(g, loc), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        val_auc = roc_auc_score(y_va.numpy(), predict(model, g_va, l_va))
        history.append({"epoch": epoch, "train_loss": total / n, "val_auc": val_auc})
        marker = ""
        if val_auc > best_auc:
            best_auc, best_state, best_epoch, wait = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
            marker = "  <- best"
        else:
            wait += 1
        print(f"{epoch:>5} {total / n:>11.4f} {val_auc:>9.3f}{marker}")
        if wait >= patience:
            print(f"No improvement for {patience} epochs; stopping early.")
            break

    model.load_state_dict(best_state)
    print(f"Keeping epoch {best_epoch} (val AUC {best_auc:.3f})")
    return model, pd.DataFrame(history), best_epoch


# ---------- Plot ----------

def plot_results(history: pd.DataFrame, best_epoch: int, y_test, probs: dict) -> None:
    fig, (ax_loss, ax_val, ax_roc) = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)

    style(ax_loss, "Training loss")
    ax_loss.grid(True, color=GRID, linewidth=0.6)
    ax_loss.plot(history["epoch"], history["train_loss"], color=BLUE, linewidth=2)
    ax_loss.set_xlabel("Epoch", color=INK, fontsize=9)

    style(ax_val, "Validation AUC (best epoch marked)")
    ax_val.grid(True, color=GRID, linewidth=0.6)
    ax_val.plot(history["epoch"], history["val_auc"], color=BLUE, linewidth=2)
    ax_val.axvline(best_epoch, color=ORANGE, linestyle="--", linewidth=1.5)
    ax_val.set_xlabel("Epoch", color=INK, fontsize=9)

    style(ax_roc, "ROC curve (test set)")
    ax_roc.grid(True, color=GRID, linewidth=0.6)
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random guess")
    for (name, prob), color in zip(probs.items(), (ORANGE, BLUE, AQUA)):
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {roc_auc_score(y_test, prob):.3f})")
    ax_roc.set_xlabel("False positive rate (fakes called planets)", color=INK, fontsize=9)
    ax_roc.set_ylabel("True positive rate (planets caught)", color=INK, fontsize=9)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN on global + local views.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many epochs without improvement")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(VIEWS_PATH)
    df = pd.DataFrame({"kepoi_name": data["kepoi_name"], "kepid": data["kepid"], "label": data["label"]})
    split = pd.read_csv(SPLIT_PATH).set_index("kepid")["split"]
    df["split"] = df["kepid"].map(split)
    if df["split"].isna().any():
        raise SystemExit(f"{df['split'].isna().sum()} KOIs have no split; re-run train_classifier.py first.")

    # Carve a validation set (by star) out of the training stars
    train_all = df.index[df["split"] == "train"].to_numpy()
    test_idx = df.index[df["split"] == "test"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fit_pos, val_pos = next(splitter.split(train_all, groups=df.loc[train_all, "kepid"]))
    fit_idx, val_idx = train_all[fit_pos], train_all[val_pos]
    print(f"Train: {len(fit_idx)} | validation: {len(val_idx)} | test: {len(test_idx)} KOIs")

    g_all, l_all = as_tensor(data["global_view"]), as_tensor(data["local_view"])
    y_all = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)

    def subset(idx):
        return g_all[idx], l_all[idx], y_all[idx]

    model, history, best_epoch = train_model(subset(fit_idx), subset(val_idx), args.epochs, args.patience)

    # Test-set evaluation, alongside the classic model on the same KOIs
    test = df.loc[test_idx, ["kepid", "kepoi_name", "label"]].copy()
    test["prob_cnn"] = predict(model, g_all[test_idx], l_all[test_idx])
    classic = pd.read_csv(CLASSIC_PREDICTIONS)[["kepoi_name", "prob_planet"]]
    both = test.merge(classic.rename(columns={"prob_planet": "prob_classic"}), on="kepoi_name")
    both["prob_ensemble"] = (both["prob_cnn"] + both["prob_classic"]) / 2

    y = both["label"].to_numpy()
    probs = {"classic": both["prob_classic"].to_numpy(), "CNN": both["prob_cnn"].to_numpy(),
             "ensemble": both["prob_ensemble"].to_numpy()}
    test_scores = {name: scores(y, p) for name, p in probs.items()}
    print_scores(f"Test set ({len(both)} KOIs scored by both models)", test_scores)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "cnn.pt")
    both.to_csv(CNN_PREDICTIONS, index=False)
    (MODEL_DIR / "cnn_metrics.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "test": test_scores}, indent=2))
    print(f"\nSaved {MODEL_DIR / 'cnn.pt'}, {CNN_PREDICTIONS}, {MODEL_DIR / 'cnn_metrics.json'}")

    plot_results(history, best_epoch, y, probs)


if __name__ == "__main__":
    main()
