# Exoplanet Hunter: technical write-up

How the pipeline works, why it's built the way it is, and where it goes wrong. For a quick overview, see the
[README](README.md).

## 1. The problem

When a planet passes in front of its star, the star dims slightly for a few hours, once per orbit. NASA's Kepler
telescope watched about 150,000 stars for four years (2009–2013) looking for these dips. Its pipeline flagged
9,564 periodic dips as **Kepler Objects of Interest (KOIs)**. Many KOIs are impostors. The most common is an
eclipsing binary: two stars orbiting each other, either directly or blended into the target star's light from
behind. Each KOI in NASA's cumulative table has a disposition:

| Disposition | Count | Used for |
|---|---|---|
| CONFIRMED | 2,748 | training/test label 1 |
| FALSE POSITIVE | 4,839 | training/test label 0 |
| CANDIDATE | 1,977 | held out entirely; scored at the end (section 9) |

The task: given a KOI's light curve and NASA's period and timing for it, predict whether it is a real planet.

## 2. Data

- **Catalog.** The NASA Exoplanet Archive `cumulative` table (via TAP): period, mid-transit time (BKJD),
  duration, depth and disposition for every KOI.
- **Usable KOIs.** Rows need a period, mid-transit time and duration, and a duration of **15 hours or less**. The
  light-curve cleaning uses a ~2-day smoothing window, which starts to absorb longer transits. 557 KOIs fail this
  check. That leaves 2,724 confirmed, 4,410 false positives and 1,873 candidates.
- **Training sample.** 500 confirmed and 500 false positives, sampled at random with a fixed seed. That's 1,000 KOIs
  on 947 stars. Balancing the classes makes metrics easy to read, but it is not the real mix (section 10).
- **Light curves.** Kepler long-cadence data (one point per 30 minutes), all available quarters, downloaded from
  MAST with lightkurve and stitched together; each quarter is normalized to ~1.0. About 6 GB for the training stars.
- **Coverage.** Features were computed for 994 of the 1,000 KOIs. Of the 6 failures, 4 were MAST download
  timeouts and 2 had no usable data near the catalog ephemeris.

## 3. Light-curve processing

All of this happens once per star, so every KOI on the star sees the same cleaned data.

1. **Clip high outliers only** (above 4σ): cosmic rays and flares. Low points are kept, because transits *are*
   low points.
2. **Flatten.** Fit the star's slow variability with a Savitzky-Golay filter over 101 cadences (~2 days) and
   divide it out. The window has to be several times the longest transit, or the filter bends into the dip and
   makes it shallower.
3. **Mask every known KOI from the trend fit.** Each KOI's transits on this star, padded to twice their duration,
   are excluded when fitting the trend, so the trend can't partly fit away a real dip.
4. **Per-KOI isolation.** Before computing one KOI's features, the *other* KOIs' transits (±1 duration) are
   removed, so a neighboring planet can't look like a secondary eclipse or an odd/even difference.

### Transit search (demonstration only)

`search_transits.py` runs a Box Least Squares (BLS) search, the classic transit finder. The trial durations
run from ~1 to ~7 hours. The period grid is log-spaced and as dense as needed so that a wrong trial period can't
let a real transit drift out of alignment by more than half the shortest duration over the full baseline. After
each detection, that signal's transits are masked and the search runs again. On Kepler-10 it recovers both known
planets: **0.83750 d** (Kepler-10 b) and **45.2944 d** (Kepler-10 c).

The classifier does **not** use this search; it uses NASA's catalog ephemerides, as AstroNet did. Running BLS on
~2,700 stars would take ~1–2 minutes each, and NASA's ephemerides are better than a single search would give.
The trade-off is in section 10: the models judge signals NASA already found; they don't find new ones.

## 4. Vetting features (classic model)

Fourteen features, each aimed at a specific way impostors give themselves away. Depths, odd/even depths and the
phase-0.5 depth come from astropy's BLS `compute_stats` at the catalog ephemeris; the rest are custom.

| Feature | What it catches |
|---|---|
| `shape_ratio` | Edge depth ÷ center depth. ~1 for a flat-bottomed "U" planet transit, ~0.4 for a "V" grazing eclipse |
| `odd_even_sigma`, `odd_even_rel` | An eclipsing binary seen at half its true period alternates deep and shallow eclipses |
| `secondary_snr`, `secondary_ratio` | A dip half an orbit later means the companion gives off light, so it's a star |
| `duration_frac` | Duration ÷ period. Large values suggest two big stars eclipsing each other |
| `harmonic_dll` | How much better a sine wave fits than a transit: stellar variability, not an eclipse |
| `max_single_frac` | Share of the signal from the strongest single event: one glitch faking a periodic signal |
| `depth_ppm`, `depth_snr`, `n_transits`, `noise_ppm`, `period_days`, `duration_hours` | Context: how strong the signal is, and how much data supports it |

**Why both a "sigma" and a "relative" version** of the odd/even and secondary tests: at very high SNR, even a 1%
difference is statistically significant. Kepler-10 b has SNR ~270, so a sigma-only test would flag bright, real
planets. The relative version measures the *size* of the difference instead.

`shape_ratio` compares the median depth within ±1/6 of a duration of mid-transit (the center) with the median
depth between 1/6 and 1/2 of a duration away (the edges). Both are measured against an out-of-transit baseline
1 to 3 durations from mid-transit.

## 5. Evaluation design

- **Split by star, not by KOI.** Planets on the same star share a light curve and noise properties, so letting
  them straddle train and test would leak information. `GroupShuffleSplit` (80/20, grouped by star) gives
  **798 training KOIs on 752 stars and 196 test KOIs on 189 stars**. The split is saved to `data/split.csv` and
  reused by every model, so comparisons are like for like.
- **Cross-validation** on the training set only: 5-fold `StratifiedGroupKFold`, also grouped by star.
- The test set is used for the final scores of fixed models (section 10 covers how it has been reused since).

## 6. Models

### Classic: gradient-boosted trees

scikit-learn `HistGradientBoostingClassifier`, deliberately small for ~800 rows: 300 iterations, learning
rate 0.05, at most 15 leaves per tree, at least 20 samples per leaf, L2 regularization 1.0. It handles missing
values natively; for example, `shape_ratio` is undefined when there are too few in-transit points.

A **baseline** using only period, duration and depth shows how much the vetting features add:

| | CV accuracy | CV AUC | Test accuracy | Test AUC |
|---|---|---|---|---|
| Baseline (3 features) | 0.805 | 0.862 | 0.821 | 0.874 |
| Full (14 features) | 0.890 | 0.963 | **0.929** | 0.962 |

### CNN: AstroNet-lite

Each KOI becomes two fixed-size "views" of its folded light curve, following Shallue & Vanderburg (2018):

- **Global view:** the whole orbit (phase −P/2 to +P/2), median-binned into 2,001 points. It shows secondary
  eclipses and anything else around the orbit.
- **Local view:** ±2 transit durations, median-binned into 201 points. It shows the transit's shape.
- Empty bins are filled by interpolation. Each view is shifted to median 0 and scaled so its lowest point is −1,
  so the network learns shapes rather than raw depths.

The architecture has two branches of 1D convolution blocks. Each block is two convolutions (kernel 5) plus max
pooling:
- **Global branch:** 4 blocks with 8 → 64 channels.
- **Local branch:** 2 blocks with 16 → 32 channels.
- **Head:** each branch is max-pooled to length 8, then the two are joined and passed through a 128-unit dense
  layer, 40% dropout and one output.

**Training:**
- **Data:** 15% of the training *stars* are held out for validation, giving 680 KOIs for fitting and 118 for
  validation.
- **Optimizer:** Adam (learning rate 1e-3, weight decay 1e-4), binary cross-entropy loss, batch size 32.
- **Augmentation:** each example is randomly flipped left-to-right, since a transit looks the same played
  backwards.
- **Early stopping:** on validation AUC with patience 15. The best epoch was 20, and training stopped at 35.

### Ensemble

The mean of the two models' probabilities.

### Results (test set, 196 KOIs)

| Model | Accuracy | Precision | Recall | ROC AUC |
|---|---|---|---|---|
| Baseline trees | 0.821 | 0.827 | 0.835 | 0.874 |
| Classic trees | **0.929** | **0.908** | **0.961** | 0.962 |
| CNN | 0.867 | 0.874 | 0.874 | 0.941 |
| Ensemble | 0.918 | 0.899 | 0.951 | **0.965** |

The classic model caught 99 of 103 planets and passed 10 of 93 false positives. The CNN trails it, which is
unsurprising: AstroNet trained on ~15,000 labeled signals, and this one saw 680. A single training run on so little
data is also noisy, so the CNN's numbers carry more uncertainty than the table suggests. The ensemble has the
best AUC (better *ranking*) but lower accuracy at the 0.5 threshold. Averaging helps when both models are
unsure, and can hurt when one is confidently wrong. On K00403.01, the CNN correctly says false positive (0.23),
the trees say planet (0.99), and the average (0.61) sides with the trees.

## 7. What the classic model relies on

Permutation importance on the test set (drop in ROC AUC when one feature is shuffled):

| Feature | AUC drop |
|---|---|
| `shape_ratio` | **0.142** |
| `duration_frac` | 0.028 |
| `depth_snr` | 0.017 |
| `n_transits` | 0.008 |
| `secondary_ratio` | 0.006 |
| odd/even features | ≤ 0.002 |

Transit shape dominates. The odd/even tests barely register, plausibly because the shape and duration features
already catch most of the eclipsing binaries they would have flagged. Correlated features share credit under
permutation importance, so "barely registers" means "adds little given the others", not "useless".

## 8. Error analysis

### The confident mistakes are blended binaries

The classic model's most confident errors are false positives it is ~99% sure are planets, such as K01262.01 and
K02969.01. The hypothesis was that these are **blended eclipsing binaries**: a background binary star whose light
mixes with the target's, producing a dip that looks like a planet transit. Brightness alone can't tell those
apart. NASA identifies them from the image: if the light dims at a slightly different position on the detector
than the target star, the "centroid offset" flag is raised.

To test this, NASA's four false-positive flags were fetched for every KOI (stored separately, so the training
catalog never changed). They were compared across the false positives the model caught and the ones it missed.
To get more than the test set's 93 false positives, each training KOI's prediction comes from **out-of-fold**
cross-validation (same 5-fold star-grouped recipe), so it was made by a model that never saw that KOI. That
gives 495 false positives in total.

| Our false positives | n | Centroid offset | Stellar eclipse | Not transit-like | Ephemeris match |
|---|---|---|---|---|---|
| Caught (p < 0.5) | 439 | 39% | 49% | 29% | 24% |
| Missed (0.5–0.9) | 30 | 57% | 20% | 30% | 13% |
| Missed, confident (p > 0.9) | 26 | **81%** | 35% | 4% | 38% |

Across all misses, 68% carry the centroid-offset flag vs. 39% of catches: **odds ratio 3.3, Fisher exact
p = 5e-5**. The stellar-eclipse flag goes the other way (27% of misses vs. 49% of catches, p = 0.003). Odd/even
differences and secondary eclipses *are* visible in a light curve, so the model catches those. Instrument and
variability glitches ("not transit-like") almost never fool it with confidence (4%).

This supports the hypothesis without proving it. The result is correlational, covers the classic model only,
and NASA's labels are themselves partly *based* on these flags (section 10). The practical conclusion holds,
though: **a bigger light-curve model won't fix these errors; centroid or pixel-level features would.**

### Other notable cases

- **K02034.02 (Kepler-1065 c)**, a confirmed planet both models reject (ensemble 0.03). Its phase-0.5 window shows
  a dip about 37% as deep as the transit (`secondary_ratio` 0.37), which looks like a glowing companion. The cause
  is unknown. It might be a real feature of the system or a processing artifact, and it wasn't investigated further.
- **K01133.01**, an eclipsing binary caught easily. Its eclipse depth rises and falls on a ~370-day cycle.
  Kepler rolled 90° every quarter, so the star landed on different pixels each season. Season-dependent depth
  is a sign of light from a neighbor, the same blending mechanism as above (unverified for this star).

## 9. The CANDIDATE KOIs

All 1,873 candidates were scored through the same code path as the dashboard (`inference.analyze`). This is the
same path verified against training in section 11. 1,871 were scored; 2 have no usable data near their transits.

| | Classic | CNN | Ensemble |
|---|---|---|---|
| Called planet | 618 (33%) | 519 (28%) | 571 (31%) |

### The selection effect

The 31% is mostly a property of the training data. Confirming a planet takes follow-up observations, which favor
strong signals, so confirmed planets skew bright. Candidates are what was left unconfirmed, and they skew faint:

| SNR band | Training KOIs | Training planet share | Candidates | Candidates called planet | Robovetter mean score |
|---|---|---|---|---|---|
| ≤ 10 | 79 | **4%** | 559 | **0%** | 0.75 |
| 10–15 | 75 | 31% | 599 | 34% | 0.86 |
| 15–20 | 114 | 67% | 236 | 62% | 0.82 |
| 20–30 | 151 | 71% | 148 | 57% | 0.83 |
| 30–50 | 171 | 76% | 105 | 55% | 0.81 |
| > 50 | 399 | 40% | 212 | 38% | 0.69 |

(Median SNR: candidates 12.4, training planets 35.8. In the > 50 band, false positives outnumber planets again;
many of them are eclipsing binaries, whose eclipses are far deeper than a planet's transit.)

In each band, the share of candidates called planets tracks the training base rate. The models learned "faint
means false positive" because that was true *of the training sample*. Every candidate with SNR ≤ 10 is
rejected, even though NASA's Robovetter (its automated vetter, which every candidate passed) scores them 0.75 on
average. A second symptom: the two models agree on 74% of candidates vs. 89% of the test set, as expected when
inputs are unlike the training data.

### Where the verdicts are usable

Above SNR 15, where training had both classes in reasonable numbers:

- The ensemble calls **370 candidates planets** (46 with both models ≥ 0.9). They are listed in
  `data/candidates_flagged.csv`.
- Where a Robovetter score exists, it agrees with **315 of our 349 planet calls (90%)**. Every one of our 15
  top-flagged candidates scores 0.86 or higher.
- The reverse agreement is weaker. The Robovetter passes 145 of the 236 strong-signal candidates we reject.
  Overall Spearman correlation between the two scores is 0.30.

These are model scores, not discoveries.

## 10. Threats to validity

- **Labels come largely from NASA's own vetting.** False-positive dispositions are heavily based on the
  Robovetter and its flags, so the models partly learn to imitate NASA's process. That's why section 8 treats the
  flags as a description of *how* NASA caught each impostor, not as independent ground truth.
- **Balanced training data.** The real mix of labeled usable KOIs is 38% planets, not 50%. The classic model's
  rates on the test set (96.1% of planets caught, 10.8% of false positives passed) would give a precision of
  about **0.85** at the real mix, not 0.908.
- **Faint-signal bias** (section 9). Training planets are mostly strong signals, so verdicts below SNR ~15 mostly
  reflect that.
- **Test-set reuse.** The test set gave the final scores of fixed models. It has since been used for importance,
  error analysis and choosing showcase examples. Future model changes should be judged by cross-validation or a
  fresh split, not by these 196 KOIs.
- **Small data, single seed.** 1,000 labeled KOIs and one CNN training run. Differences of a point or two
  between models are within the noise.
- **Brightness only.** No pixel or centroid information (section 8).
- **Long transits** (> 15 h) were excluded from training; the dashboard warns when one is entered.

## 11. Engineering

### One code path, verified

The dashboard and the candidate batch both call `inference.analyze`. It uses the same functions that built
the training data: `prepare_star`, `koi_data`, `extract_features` and `make_views`. `python inference.py --parity`
re-scores all 196 test KOIs through that path and compares with the saved training-time results:
- **Features:** identical.
- **Probabilities:** classic within 1e-16, CNN within 2e-7.
- **Verdicts:** none flip.

It was re-run after each change to the pipeline. The showcase data (`build_demo.py`) is checked the same way
before it's saved.

### Dashboard

Streamlit, with all the science in the UI-agnostic `inference.py` (so an API layer could be added without a
rewrite) and Plotly charts in `figures.py`.

**Data:** stars load from the local cache, or are downloaded live. Ten showcase stars ship as cleaned arrays
(full float64, ~1 MB each) so they load instantly without the 6 GB cache.

**Slow downloads.** astroquery's file downloads have **no timeout**, so a stalled MAST connection would hang
the page forever. Each analysis therefore runs in a worker thread, and the page waits at most 90 s. A slow job
keeps running in the background and is picked up on retry; failed jobs are retried rather than cached.

### Bugs found along the way

- **lightkurve breaks `print` under threads.** `download_all` silences astroquery by pointing `sys.stdout` at an
  `os.devnull` file and closing it afterwards. With parallel downloads, one thread can "restore" another thread's
  already-closed file, leaving `sys.stdout` permanently closed.
  - The first pilot of the candidate batch crashed this way. The resulting `ValueError` was also mislogged as
    "not enough data".
  - **Fix:** call the undecorated method (`download_all.__wrapped__`) and quiet astroquery through its logger;
    it only draws progress bars at INFO level or below.
- **Truncated cache files are never repaired.** lightkurve skips any file that already exists, so a download
  interrupted mid-file breaks that star for good. **Fix:** on a read failure, delete the unreadable files and
  fetch them again (verified on a real truncated file).
- **Timeout ambiguity.** On Python 3.11, `concurrent.futures.TimeoutError` *is* the built-in `TimeoutError`, the
  same class MAST raises. The app checks whether the job finished to tell "we stopped waiting" from "MAST timed
  out".
- **astroquery's search timeout** (600 s) is read once at import, so it is set directly on the live connection
  object (90 s).

### Batch scoring

`score_candidates.py`:
- runs 4 stars in parallel, each in a daemon thread, and abandons a star after 5 minutes so a hung download
  can't stall the run
- appends results after every star, so it's resumable
- processes stars in a fixed random order, so a partial run is still a fair sample

The full run took about 2.5 hours. 63 KOIs failed on network errors; on retry (`--retry-failed`), 62 succeeded
and 1 turned out to have no usable data.

## 12. Future work

- **Centroid or pixel-level features** (from Kepler's target pixel files) to catch blended binaries. This
  targets the model's main failure mode directly.
- **More labeled data.** About 6,100 more usable labeled KOIs weren't sampled. Training on them, weighted
  toward the real class mix, should help the CNN most and may soften the faint-signal bias, though faint
  confirmed planets will stay rare.
- **Several CNN seeds**, averaged, plus odd/even and secondary-eclipse views as in later AstroNet variants.
- **Calibration and thresholds** tuned for the real class mix; a weighted or stacked ensemble instead of a
  plain mean.
- **Multiplicity.** Candidates on stars with several signals are known to be real planets more often; the models
  don't use this at all.
- **Hosting** the dashboard (e.g. Streamlit Community Cloud or Hugging Face Spaces).

## References

- Shallue, C. J. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning: A Five-planet Resonant
  Chain around Kepler-80 and an Eighth Planet around Kepler-90.* The Astronomical Journal, 155, 94.
- NASA Exoplanet Archive, Kepler Objects of Interest cumulative table.
- Lightkurve Collaboration (2018). *Lightkurve: Kepler and TESS time series analysis in Python.*
