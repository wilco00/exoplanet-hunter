# Exoplanet Hunter

Portfolio project: an AI pipeline that classifies NASA Kepler transit signals as real
planets or false positives, from raw light curves to a public web dashboard.
Owner: Will (UGA CS, Dec 2026). Windows, PowerShell, Python venv at `.venv/`.

## Repository

- **Public on GitHub:** https://github.com/wilco00/exoplanet-hunter. Branch `main` tracks `origin/main`
  (renamed from `master` before the first push). Anything committed is public; check for secrets and personal
  data before pushing.
- **Remote URL pins the account:** `https://wilco00@github.com/wilco00/exoplanet-hunter.git`. Git Credential
  Manager also holds a login for a different GitHub account that can't push here (403). Keep the
  `wilco00@` in the URL; don't sign out or replace the other account's credential.
- **Commit email** is the GitHub noreply address, set in this repo's git config
  (`143106821+wilco00@users.noreply.github.com`), so the personal Gmail never appears in public history.
  History was rewritten once before the first push to remove it.
- **Don't rewrite pushed history** (no amend/rebase/filter-branch/force-push on `main`); add new commits instead.

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
| 7 | Web dashboard: enter a KIC/KOI, see the light curve, fold and both models' verdicts | Done: Streamlit, runs locally |
| 8 | Polish: README/write-up, run models on the 1,873 held-out CANDIDATE KOIs ("what my model flagged") | Done: 1,871 scored; README + WRITEUP.md; not hosted |

**Milestone 8 scope (agreed with Will):** score all candidates (`score_candidates.py`), compare with NASA's
Robovetter (`robovetter.py`, `analyze_candidates.py`), README + technical WRITEUP.md. Hosting was deferred
("local only for now"); no model changes, so reported numbers are unchanged. Not done: candidates page in the
dashboard, CNN seed averaging, Plotly vetting plot.

**Milestone 7 scope (agreed with Will):** Streamlit, chosen over FastAPI + frontend for speed. The science lives in a
UI-agnostic `inference.py`, so a FastAPI layer could be added later without a rewrite. Input is any catalog KOI
(catalog ephemeris, same as training); arbitrary KICs with no KOI (our own BLS) were left out. Data comes from the
local cache, else a live MAST download, plus a precomputed showcase set. Page sections: verdict cards, dataset
badge with true label, light curve, fold, "why this verdict" feature table, vetting plot.

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
  (K01262.01, K02969.01, K00403.01, ...). **NASA's flags support the blended-binary explanation** (`robovetter.py`,
  out-of-fold predictions on all 495 labeled FPs): 81% of confident misses (p > 0.9) carry the centroid-offset flag
  vs. 39% of caught FPs (all misses 68%; odds ratio 3.3, Fisher p = 5e-5). The stellar-eclipse flag goes the
  other way (27% vs. 49%). Correlational, classic model only, and NASA's labels are partly *based* on these flags.
  Fix needs centroid/pixel features, not a bigger light-curve model.
- **Precision at the real class mix is ~0.85**, not 0.908: training is balanced 50/50, but 38% of labeled usable
  KOIs are planets (classic test rates: 96.1% of planets caught, 10.8% of FPs passed).
- Permutation importance (test AUC drop): `shape_ratio` 0.142, `duration_frac` 0.028, `depth_snr` 0.017,
  `n_transits` 0.008, `secondary_ratio` 0.006; the odd/even features are ~0.002 or less.
- The ensemble can be worse than its better half: on K00403.01 the CNN is right (0.23) but the ensemble says
  planet (0.61).
- K01133.01's eclipse depth rises and falls on a ~370-day cycle (Kepler's seasonal rolls put the star on
  different pixels), which hints at light from a neighboring star. Unverified.

## Candidate results (milestone 8)

- 1,871 of 1,873 scored (K01101.02 and K05930.01: no usable data near the transits). Ensemble calls 571 (31%)
  planets; classic 618, CNN 519.
- **The 31% is mostly a training selection effect.** Confirmed planets skew to strong signals, so only 3 of 79
  training KOIs with SNR ≤ 10 were planets. The models reject all 559 candidates with SNR ≤ 10 (Robovetter mean
  0.75). Candidates' median SNR is 12.4 vs. 35.8 for training planets. Per SNR band, the share of candidates called
  planets tracks the training base rate. Classic and CNN agree on 74% of candidates vs. 89% of the test set.
- **Trust verdicts only above SNR ~15.** There the ensemble flags 370 (46 with both models ≥ 0.9;
  `data/candidates_flagged.csv`). The Robovetter agrees with 315 of our 349 scored planet calls (90%), but also
  passes 145 of the 236 strong-signal candidates we reject. Overall Spearman with the Robovetter score is 0.30.
- Always present the candidates as model scores, not discoveries, and lead with the SNR caveat.

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
| `inference.py` | Dashboard engine: `find_kois(query)` (KOI / KIC / Kepler name), `analyze(koi)` -> `KoiResult` (same path as training), `dataset_of` (train/test/unseen/candidate), `feature_context` + `feature_importance`. CLI: `python inference.py Kepler-10`; `--parity` re-scores the 196 test KOIs and must match saved results exactly | — |
| `build_demo.py` | Saves the 10 showcase KOIs' cleaned light curves; verifies they re-score identically | `data/demo/kic<KIC>.npz`, `data/demo/showcase.csv` |
| `app.py` + `figures.py` | Streamlit page (`figures.py` = Plotly light curve + fold). Run `.venv\Scripts\python.exe -m streamlit run app.py`, or the `dashboard` entry in `.claude/launch.json` (port 8501). `?koi=K01262.01` opens a KOI | — |
| `score_candidates.py` | Scores `candidates.csv` via `inference.analyze`; 4 parallel daemon-thread workers, stars abandoned after `--timeout` (300 s), **resumable**, fixed random star order; `--retry-failed`, `--limit` (~2.5 h full run) | `data/candidate_predictions.csv`, `data/candidate_failed.csv` |
| `robovetter.py` | Fetches NASA's `koi_score` + 4 FP flags into a separate file; out-of-fold classic predictions on all labeled KOIs; caught-vs-missed flag analysis (Fisher) | `data/koi_vetting_flags.csv`, `plots/mistakes_vs_nasa_flags.png` |
| `analyze_candidates.py` | Candidate verdicts by SNR band vs. training base rate, Robovetter agreement, flagged list (SNR > 15) | `data/candidates_flagged.csv`, `plots/candidates_snr.png` |

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
  The dashboard uses its light **and** dark steps (dark: blue `#3987e5`, orange `#d95926`), following Streamlit's theme.
- **Dashboard = training pipeline, verified.** `inference.analyze` reuses `prepare_star` / `koi_data` /
  `extract_features` / `make_views`. `python inference.py --parity` must stay exact (it is: features identical,
  classic 1e-16, CNN 2e-7). Re-run it after touching any of those.
- **Dataset badge has four cases:** train, test, *unseen* (one of ~6,500 labeled KOIs outside our 1,000, so a
  fair out-of-sample check), candidate. An unseen KOI on a training star gets a note (the models saw that star).
- **"Why this verdict"** = per feature, the share of planets among the 50 nearest training KOIs (excluding itself),
  rows sorted by permutation importance computed at startup. One feature at a time, not a model attribution.
- **Showcase data is full float64** (~1 MB/star, 8.9 MB total) so demo results match the full pipeline exactly.
- **Analyses run in a worker thread with a 90 s wait** (`app.run_analysis`). Slow jobs keep running and are picked
  up on retry; failed jobs are retried rather than cached.

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
  candidate_predictions.csv 1,871 scored candidates: probabilities + features + warnings (score_candidates.py)
  candidate_failed.csv      the 2 candidates with no usable data
  candidates_flagged.csv    370 candidates the ensemble calls planets with SNR > 15 (analyze_candidates.py)
  koi_vetting_flags.csv     NASA koi_score + 4 FP flags for every KOI (robovetter.py; separate so the catalog never changes)
  lightkurve_cache/         ~17 GB raw FITS (training ~6 GB + candidates), mastDownload/Kepler/kplr<9-digit KIC>_lc_*/ *_llc.fits  (gitignored)
  views/, views.npz         CNN inputs (gitignored, regenerable in minutes)
  demo/                     showcase stars' cleaned light curves + showcase.csv (tracked; build_demo.py)
models/                     classic_full.joblib, cnn.pt, *_metrics.json (1.2 MB, tracked so a clone runs the dashboard)
plots/                      all figures (tracked; used by README/WRITEUP)
docs/                       README screenshots (headless Edge captures, see Gotchas)
README.md, WRITEUP.md       recruiter-facing overview / long technical write-up; keep their numbers in sync with this file
.streamlit/config.toml      dashboard accent color, per light/dark theme
.claude/launch.json         `dashboard` server entry (runOnSave on)
```

## Gotchas

- **MAST timeouts:** downloads occasionally hang for the full 600 s astroquery timeout. 4 of the 6 failures were
  timeouts (retryable with `python build_features.py --retry-failed`); 2 had no data near the catalog ephemeris.
  astroquery reads its *search* timeout once at import, so `inference.py` sets
  `Observations._portal_api_connection.TIMEOUT = 90` directly. *File downloads* have no timeout at all
  (`timeout=None`), which is why the app wraps analyses in a timed worker thread.
- **Python 3.11: `concurrent.futures.TimeoutError is TimeoutError`** (an `OSError`), the same class MAST raises.
  `app.run_analysis` checks `job.done()` to tell "we stopped waiting" from "MAST timed out".
- **Retrying failures changes the star set:** newly recovered stars aren't in `split.csv`, and
  `train_classifier.py` will then make a NEW split (changing all reported numbers). If you retry, assign the
  new stars to `train` in `split.csv` rather than regenerating it.
- **Windows file locks:** `savefig` fails with `OSError [Errno 22] Invalid argument` if the PNG is open in a
  viewer. Close it and re-run (resumable scripts skip finished work).
- **lightkurve breaks `print` under threads.** `SearchResult.download_all` is wrapped in `suppress_stdout`, which
  points `sys.stdout` at an `os.devnull` file and closes it afterwards. With parallel downloads, threads restore
  each other's closed file, so every later `print` fails with `ValueError: I/O operation on closed file` (which
  then looks like a data error). `fetch(verbose=False)` avoids it: it calls `download_all.__wrapped__` and sets
  astroquery's logger to WARNING (astroquery only draws progress bars at INFO). Use `verbose=False` in any
  threaded code. `build_features.py` still uses `redirect_stdout` (single-threaded, fine).
  `fetch()` raises `LookupError` (no data) and `ConnectionError` (download failed), not `SystemExit`.
- **Truncated cache files:** an interrupted download (Ctrl+C, abandoned timeout thread) leaves a short FITS file,
  and lightkurve never re-downloads a file that already exists. `inference.load_light_curve` deletes unreadable
  cached files (`remove_unreadable`) and fetches again. `build_views.load_cached` alone does not.
- **Test set has been reused** for importance, error analysis and showcase picks. Judge future model changes by
  grouped CV or a fresh split, not by the 196 test KOIs.
- **Screenshots:** the browser pane caps captures at 800 px. For README images, use headless Edge
  (`msedge --headless=new --user-data-dir=<scratch dir> --window-size=1400,H --force-device-scale-factor=1.5
  --virtual-time-budget=25000 --screenshot=out.png http://localhost:8501/?koi=...`). It needs its own profile dir
  and a warm server (the first request after a restart computes feature importance and may not render in time).
- **Streamlit:** restart the server after changing `.streamlit/config.toml`. A plain `[theme] primaryColor`
  pins the app to light mode; use `[theme.light]` / `[theme.dark]`. Plotly WebGL traces (`Scattergl`) draw above
  SVG traces, so a line meant to sit on top of WebGL points must also be `Scattergl`.
- **Rebuild the showcase** (`python build_demo.py`) after changing the cleaning step or refreshing the catalog:
  `cleaned_star` prefers `data/demo/` files over the cache, so stale ones would silently win.
- `python extract_features.py --koi ...` re-saves a tracked PNG in `plots/` (output is deterministic, so no diff).
- lightkurve emits an "oktopus" warning on import; it's harmless and filtered in each script.
- Known cosmetic issues for polish: in `*_bls.png` the dashed best-period line hides signal 2's peak, and
  outliers stretch the y-axis on the 10b fold. In the dashboard's dark mode the matplotlib vetting plot stays a
  light panel (port it to Plotly to fix).

## Ideas noted for later

- CNN: average several seeds (single-seed scores are noisy); add odd/even and secondary-eclipse views (as in
  later AstroNet variants); tune threshold on the validation set.
- **Centroid / pixel-level features** (Kepler target pixel files) to catch blended binaries, the main failure mode.
- **More labeled data:** ~6,100 more usable labeled KOIs weren't sampled; train on the real class mix. May soften
  the faint-signal bias (faint confirmed planets stay rare). Calibrate/threshold for the 38% planet mix.
- **Multiplicity:** candidates on multi-KOI stars are more often real; the models don't use it.
- **Hosting** (deferred in milestone 8): Streamlit Community Cloud or Hugging Face Spaces. `models/` and
  `data/demo/` are committed; showcase KOIs work without the FITS cache, anything else downloads live.
  Check torch + lightkurve fit the host's memory.
- Dashboard: a "what my model flagged" candidates page; SHAP per-prediction attributions; arbitrary KICs via our
  own BLS search; a weighted or stacked ensemble instead of a plain mean; Plotly vetting plot (dark mode).
