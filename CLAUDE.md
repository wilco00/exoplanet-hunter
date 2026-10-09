# Exoplanet Hunter

Portfolio project: an AI pipeline that classifies NASA Kepler transit signals as real
planets or false positives, from raw light curves to a public web dashboard.
Owner: Will (UGA CS, Dec 2026). Windows, PowerShell, Python venv at `.venv/`.

## How Will wants to work (follow these)

- Ask scope questions before starting anything new; present a plan and get approval before building.
- Build **one system at a time** in small, compiling increments. Don't generate many files in one pass.
- After each step, **run it yourself**, check the output and plots, then say exactly what to look at and stop.
- Explain results plainly (Will is new to astronomy; ML/CS background is solid).
- If a documented skill or tool applies, say so rather than improvising around it.
- Be honest when a prediction was wrong. Don't oversell results.

## Milestone plan and status

| # | Milestone | Status |
|---|---|---|
| 1 | Fetch and plot one light curve (Kepler-10) | Done |
| 2 | Clean: clip high outliers, flatten with Savitzky-Golay (~2-day window) | Done |
| 3 | BLS transit search | Done: recovered Kepler-10b (0.83750 d) and 10c (45.2944 d) |
| 4a | Labeled KOI list: 500 confirmed + 500 false positives | Done |
| 4b | Vetting features for one KOI + vetting plot | Done |
| 4c | Batch features for all training KOIs | Done: 994/1000 (6 failed, see below) |
| 5 | Classic classifier (gradient-boosted trees) | Done |
| 6a | CNN input views (global 2001 + local 201 bins) | Done: 994 KOIs |
| 6b | 1D CNN (AstroNet-lite, PyTorch) + ensemble | Done (first run). Optional tuning next |
| 7 | Web dashboard: enter a KIC/KOI, see the light curve, fold and both models' verdicts | **Next** |
| 8 | Polish: README/write-up, deploy, run models on the 1,873 held-out CANDIDATE KOIs ("what my model flagged") | Not started |

**Open decision for milestone 7:** Streamlit (all Python, fastest) vs. FastAPI + separate
frontend (more work, stronger software-engineering signal). Ask Will before building.

## Results so far (test set: 196 KOIs on 189 stars, never used in training)

| Model | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Baseline trees (period, duration, depth only) | 0.821 | 0.827 | 0.835 | 0.874 |
| Classic trees (all vetting features) | **0.929** | 0.908 | 0.961 | 0.962 |
| CNN (views only) | 0.867 | 0.874 | 0.874 | 0.941 |
| Ensemble (mean of classic + CNN) | 0.918 | 0.899 | 0.951 | **0.965** |

- `shape_ratio` (U vs. V transit shape) is by far the most important classic feature.
- The CNN's best validation AUC was at epoch 20 of 35 (early stopping, patience 15). It's a single
  seed on ~680 training examples (AstroNet had ~15k), so the scores are noisy.
- Most confident classic mistakes are false positives the model is ~99% sure are planets
  (K01262.01, K02969.01, K00403.01, ...). The likely cause is blended background eclipsing binaries, which
  can't be seen from flux alone (they need centroid/pixel data). This is unconfirmed; worth stating as a limitation.

## Scripts (run from the project root, in this order)

| Script | What it does | Output |
|---|---|---|
| `fetch_lightcurve.py` | `fetch(target, max_quarters, download_dir)`: MAST search + download + stitch | plots for one star |
| `clean_lightcurve.py` | `clean(lc, window_length, sigma_upper, ephemeris)`: clip high outliers, flatten. `ephemeris` = one `(P, t0, dur_days)` or a list; those transits are masked from the trend fit | plot |
| `search_transits.py` | BLS search, iterative masking for multiple planets; `fold_hours`, `binned_median` helpers | `plots/<target>_bls.png` |
| `build_koi_list.py` | Downloads the NASA Exoplanet Archive `cumulative` KOI table (TAP), builds a balanced training list | `data/koi_cumulative.csv`, `data/training_set.csv`, `data/candidates.csv` |
| `extract_features.py` | `star_ephemerides`, `prepare_star` (clean once per star, all KOIs masked), `koi_data` (remove *other* KOIs' transits), `extract_features` (astropy BLS `compute_stats` + custom shape/ratio features). CLI: `--koi K00072.01` makes a vetting plot | `plots/<KOI>_vetting.png` |
| `build_features.py` | Batch over `training_set.csv`, one download per star, **resumable**, logs failures | `data/features.csv`, `data/failed.csv` |
| `train_classifier.py` | Star-level 80/20 split (saved), 5-fold grouped CV, baseline vs. full HistGradientBoosting, permutation importance | `data/split.csv`, `models/classic_full.joblib`, `data/test_predictions_classic.csv`, `plots/classic_results.png` |
| `build_views.py` | Global/local views from the **local cache** (no network), resumable | `data/views/*.npz`, `data/views.npz`, `plots/view_examples.png` |
| `train_cnn.py` | Two-branch 1D CNN, flip augmentation, early stopping on a 15% validation split of training stars; scores classic/CNN/ensemble on the same test KOIs | `models/cnn.pt`, `data/test_predictions_cnn.csv`, `plots/cnn_results.png` |

## Key design decisions (and why)

- **Catalog ephemerides, not our own BLS, for training data.** NASA's KOI table already has period, t0 (BKJD)
  and duration per KOI. Running BLS per star would take ~1-2 min each. AstroNet did the same.
- **Train only on CONFIRMED vs. FALSE POSITIVE.** CANDIDATEs are held out entirely for milestone 8.
- **Split by star, not by KOI** (multi-planet stars must not straddle train/test). `data/split.csv` is the
  single source of truth; the CNN reuses it so comparisons are fair. Don't regenerate it casually.
- **Both sigma and relative versions** of the odd/even and secondary features: at very high SNR (e.g. Kepler-10b,
  SNR ~270) a 1% difference is "significant", so sigma-only tests misfire on bright real planets.
- **Mask other KOIs** on the same star from both the trend fit and the per-KOI features.
- Plots use the dataviz skill's reference palette (blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, neutral ink).

## Data layout

```
data/
  koi_cumulative.csv        full KOI catalog (9,564 rows)
  training_set.csv          1,000 labeled KOIs (947 stars)
  candidates.csv            1,873 held-out CANDIDATE KOIs (milestone 8)
  features.csv              994 rows of vetting features + label
  failed.csv                KOIs that failed in build_features.py
  split.csv                 kepid -> train/test (DO NOT regenerate casually)
  test_predictions_*.csv    per-KOI test probabilities
  lightkurve_cache/         ~6 GB raw FITS, mastDownload/Kepler/kplr<9-digit KIC>_lc_*/ *_llc.fits  (gitignored)
  views/, views.npz         CNN inputs (gitignored, regenerable in minutes)
models/                     classic_full.joblib, cnn.pt, *_metrics.json (gitignored)
plots/                      all figures (tracked; useful for the README)
```

## Gotchas

- **MAST timeouts:** downloads occasionally hang for the full 600 s astroquery timeout. 4 of the 6 failures were
  timeouts (retryable with `python build_features.py --retry-failed`); 2 had no data near the catalog ephemeris.
  If timeouts become frequent, lower astroquery's MAST timeout instead of waiting 10 min per stall.
- **Retrying failures changes the star set:** newly recovered stars aren't in `split.csv`, and
  `train_classifier.py` will then make a NEW split (changing all reported numbers). If you retry, assign the
  new stars to `train` in `split.csv` rather than regenerating it.
- **Windows file locks:** `savefig` fails with `OSError [Errno 22] Invalid argument` if the PNG is open in a
  viewer. Close it and re-run (resumable scripts skip finished work).
- `fetch()` prints progress; `build_features.py` silences it with `redirect_stdout`.
- lightkurve emits an "oktopus" warning on import; it's harmless and filtered in each script.
- Known cosmetic issues for polish: in `*_bls.png` the dashed best-period line hides signal 2's peak, and
  outliers stretch the y-axis on the 10b fold.

## Ideas noted for later

- CNN: average several seeds (single-seed scores are noisy); add odd/even and secondary-eclipse views (as in
  later AstroNet variants); tune threshold on the validation set.
- Milestone 8 needs features and views for `candidates.csv`: generalize `build_features.py`/`build_views.py`
  to take a KOI list (~1,800 more stars to download, likely another overnight run).
