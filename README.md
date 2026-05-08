# Flight Delay Predictor

Predicting EC261-eligible flight delays as a decision-under-uncertainty problem.
Final project for *Machine Learning Foundations* (BCSAI2025CSAI.2.M.A C2 493615) — IE University, Spring 2026.

## TL;DR

We treat flight-delay prediction not as a classification accuracy problem but as a
**positive-expected-value betting problem under EC261 (EU 261/2004) compensation rules**.
The headline plot is a *profit-versus-threshold* curve, not an F1 score. The model is
trained on US BTS data (~42M flights) using EC261-equivalent labels, then transferred
to EUROCONTROL ADRR data for an EU validation case study.

## Why this is interesting

- **The decision threshold is flight-specific.** A €30 short-haul ticket needs a much
  higher predicted P(delay) to be worth buying than a €300 long-haul ticket, because
  the EC261 payout (€250–€600) is a step function of distance. This pushes the
  problem out of standard "tune one global threshold" territory into proper
  decision-theoretic optimisation.
- **EC261's "extraordinary circumstances" exemption changes the label.** Weather,
  ATC restrictions, and strikes are exempt from compensation under the September
  2025 amendment. The right target is therefore *carrier-attributable* delay, not
  raw delay — which we encode using BTS cause-codes (`CARRIER_DELAY` and
  `LATE_AIRCRAFT_DELAY` only).
- **Calibration matters.** Expected-value math fails on uncalibrated probabilities,
  so every model is wrapped with isotonic calibration and we publish a reliability
  diagram per model.

## Repository layout

```
flight-profit/
  README.md
  requirements.txt
  Makefile                    # Common entry points: setup, smoke, train, report
  src/
    config.py                 # Paths and constants
    data/
      loaders.py              # BTS, ADRR, FAA loaders (parquet-first)
      bts_schema.py           # BTS PREZIP CSV → internal schema mapping
      ec261.py                # Compensation tiers + label generator
      synthetic.py            # Realistic BTS-like generator for smoke tests
    features/
      booking_time.py         # BookingTimeFeatureBuilder (leakage-safe)
      historical.py           # Train-fold-only rolling stats
      cyclical.py             # Cyclical encoders for hour/day/month
    models/
      registry.py             # Model factory: dummy, logreg, tree, RF, XGB, MLP
      calibrated.py           # CalibratedClassifierCV wrappers
    pipeline/
      build.py                # Sklearn Pipeline + ColumnTransformer assembly
      splits.py               # Year-based and TimeSeriesSplit helpers
    eval/
      profit_metric.py        # Custom Sklearn scorer (corrected cost matrix)
      threshold.py            # tau*(T, d) and global profit-curve sweep
      bootstrap.py            # Bootstrapped CIs on profit
      calibration.py          # Reliability diagrams
      shap_utils.py           # SHAP wrappers
      failure_modes.py        # Error-binning and case-study extractors
    report/
      narrative.py            # Helpers used by 99_final_pipeline notebook
  scripts/
    download_bts.py
    download_faa.py
    generate_synthetic.py
    smoke_test.py             # End-to-end on synthetic data, ~30s
  notebooks/
    00_data_acquisition.ipynb
    01_eda.ipynb
    02_feature_engineering.ipynb
    03_modeling.ipynb
    04_threshold_optimization.ipynb
    05_interpretation_shap.ipynb
    06_eu_transfer_validation.ipynb
    99_final_pipeline.ipynb   # Runs the entire pipeline top-to-bottom
  tests/
    test_ec261.py
    test_features.py
    test_profit_metric.py
    test_threshold.py
    test_pipeline.py
  reports/
    final_report.md           # 2000-word writeup (rendered to PDF for submission)
    poster_outline.md
    slides_outline.md
  data/                       # Gitignored except sample/
    raw/
    interim/
    processed/
    sample/                   # Tiny CSV checked in for CI smoke test
  .github/workflows/ci.yml    # Runs smoke_test.py on every push
```

## Quick start

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Smoke test (auto-detects real CSVs/parquets, falls back to synthetic, ~30s)
python scripts/smoke_test.py

# 3. Real data (incremental, resumable)
#    Each month is written to data/raw/bts_YYYY_MM.parquet (~8 MB each).
#    Re-running the script skips months already on disk.
python scripts/download_bts.py --years 2024                   # ~300 MB download, ~25 min
python scripts/download_bts.py --years 2023 2024              # validation + test
python scripts/download_bts.py --years 2018 2019 2020 2021 2022 2023 2024  # full plan
python scripts/download_faa.py                                # optional aircraft join

# 4. Run the full pipeline
jupyter nbconvert --to notebook --execute notebooks/99_final_pipeline.ipynb
```

The loader (`src/data/loaders.py::load_bts`) auto-prefers per-month parquet
files in `data/raw/`, falls back to per-year CSV, and finally falls back to
the synthetic generator if nothing is on disk. This means the pipeline runs
top-to-bottom in any environment — required by the rubric — and the same
notebook produces real-data results the moment the download completes.

### Storage footprint

| Source / granularity | Disk (parquet) | Disk (CSV equivalent) |
|---|---|---|
| BTS — 1 month | ~8 MB | ~250 MB |
| BTS — 1 year | ~95 MB | ~3 GB |
| BTS — 7 years (full plan) | ~670 MB | ~21 GB |
| EUROCONTROL R&D Archive — 5 months / 3.89 M EU flights | **113 MB** | ~590 MB (gzip) |

Parquet is ~30× smaller than the BTS-native CSV (and ~5× smaller than the
gzip-CSV the EUROCONTROL R&D Archive ships) thanks to typed columns and
columnar compression, and pandas reads it ~5× faster.

## Data sources

| Source | Role | Access |
|---|---|---|
| **BTS Reporting Carrier On-Time Performance** | Primary training data — per-flight schedule and actual times | Free CSV at https://www.transtats.bts.gov/Tables.asp?DB_ID=120 |
| **BTS Cause of Delay** | Label generator — minutes attributed to each delay cause | Free CSV at https://www.transtats.bts.gov/ot_delay/ot_delaycause1.asp |
| **FAA Aircraft Registry** | Aircraft age/type via tail-number join | Free CSV at https://registry.faa.gov/aircraftinquiry |
| **EUROCONTROL R&D Data Archive** | EU transfer-validation case study (5 months, 3.89 M flights) | Free, no login required, at https://www.eurocontrol.int/dashboard/rnd-data-archive |
| **EUROCONTROL ADRR** | EU transfer-validation case study (gated, full cause codes) | Free academic access at https://ext.eurocontrol.int/prisme_data_provision_hmi/ |
| **NOAA GFS reanalysis (24h forecast)** | Booking-time weather features | Free at https://www.ncei.noaa.gov |
| **OpenFlights** | Airport metadata for distance-tier mapping | Public domain CSV |

## EU validation strategy

The legal regime modelled throughout the project is EU Regulation 261/2004
(EC261). The training substrate is **US BTS** because it is the only large,
free, per-flight dataset that exposes BTS-style cause-of-delay codes —
without those, the EC261-eligible label collapses to raw "3+ hour delay" and
the project loses its main intellectual contribution (separating
carrier-attributable delays from extraordinary circumstances).

The transfer-validation chapter (`notebooks/06_eu_transfer_validation.ipynb`)
evaluates the BTS-trained model on real EU operations to measure how well the
ranking transfers. The substrate is the **EUROCONTROL R&D Data Archive** — a
free, login-free per-flight feed:

| Source | Used | Coverage | Cause codes | Notes |
|---|---|---|---|---|
| **EUROCONTROL R&D Archive** | ✅ 5 months ingested | 2023-Mar, Jun, Sep, Dec + 2024-Mar | None (filed-vs-actual times only) | Free, no login. **3.89 M real EU flights** in cache. |
| **EUROCONTROL ADRR** | future work | Europe, monthly | Coarse delay-reason groups | Requires academic registration (~24-48h human approval). |
| **OpenSky Network** | not used | Global, real-time historical | None (tracking only) | Free fallback if R&D Archive is unavailable. |

Two practical caveats — quantified in `reports/eu_data_analysis.md`:

1. **R&D Archive `FILED ARRIVAL TIME` is the latest IFPS trajectory estimate**,
   not the airline's published schedule, so ATFM slot delays are absorbed
   pre-takeoff and the `≥3h` rate sits at **0.02 %** rather than the ~3 %
   EUROCONTROL CODA reports against the published timetable. The transfer
   chapter therefore evaluates ranking quality at the **CODA-aligned `≥15min`
   threshold** (EU base rate **19.68 %**) and uses the `≥3h` rate as a
   tail-only sanity check.
2. **No cause codes** ⇒ the label is `y_eu = (arr_delay >= threshold)`, not
   the EC261-eligible label used at training time. The chapter reports
   ranking-quality metrics (decile monotonicity, top-k lift, Spearman rank
   correlation) rather than direct ROI.

Reproducing the EU pipeline takes one command after dropping the zip files at
the repo root:

```bash
unzip -q '20*.zip'                       # 5 monthly drops -> 5 directories
python scripts/process_eurocontrol.py    # ~17 s, writes data/raw/eurocontrol_*.parquet
python scripts/analyse_eurocontrol.py    # ~10 s, writes reports/eurocontrol_summary.{json,txt}
```

## Reproducing the results

```bash
make setup              # Install requirements
make smoke              # ~30s synthetic-data sanity check
make data               # Download all real data (~2-3 GB)
make train              # Train all models with hyperparameter search (~6h on laptop)
make evaluate           # Threshold sweep + SHAP + EU validation
make report             # Render 99_final_pipeline.ipynb to HTML/PDF
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs `make smoke` on every
push so the pipeline cannot silently break.

## Team and contributions

See `reports/final_report.md` for the contribution matrix. Every group member is
expected to commit to GitHub regularly — the syllabus penalises uneven commit history.

## References

- Regulation (EC) No 261/2004, as amended September 2025
- ECJ *Sturgeon* (C-402/07) — establishes 3-hour arrival-delay trigger
- AirHelp, *Annual Compensation Report* — empirical claim-success rate
- Lundberg & Lee 2017, *A Unified Approach to Interpreting Model Predictions* (SHAP)
- BTS Aviation Support Tables, *Reporting Carrier On-Time Performance* documentation
- EUROCONTROL, *R&D Data Archive — metadata and column dictionary*, accessed via https://www.eurocontrol.int/dashboard/rnd-data-archive
- EUROCONTROL, *CODA Punctuality and Delay Analysis*, monthly reports — used as the gate-to-gate reference for the FILED-vs-published comparison
