# Flight-Delay Prediction Under EC261 — A Diagnosed Negative Result

Final project for *Machine Learning Foundations* (BCSAI2025CSAI.2.M.A C2 493615) —
IE University, Spring 2026.

## TL;DR

We treat flight-delay prediction not as a classification-accuracy problem but as
a **positive-expected-value betting problem under EC261** (EU 261/2004)
compensation rules, and we show — rigorously — that **no positive-EV ticket
exists**: every calibrated model correctly learns to *abstain* (test ROI 0 %).
This is a deliberately presented, fully diagnosed **negative result**, which the
assignment brief explicitly values. The model is trained on **real US BTS 2024**
data (a seeded 150 k stratified sample of 6.97 M flights) and the conclusion is
independently confirmed on **3.89 M real EUROCONTROL flights**.

## Why it is interesting

- **The decision threshold is flight-specific.** A €30 short-haul ticket needs a
  far higher predicted P(delay) to be worth buying than a €300 long-haul one,
  because the EC261 payout (€250–600) is a step function of distance. We derive
  the per-flight optimal threshold τ\*(T, d) in closed form.
- **The negative result is structural, not a tuning artifact.** A break-even
  analysis shows the required confidence (~63 %) is ~8× the highest eligible-
  delay rate of any route-carrier cohort (7.7 %). EC261's capped payout cannot
  cover ticket + friction for a ~1 % event.
- **Calibration is decisive.** Uncalibrated models actively bet and lose money;
  isotonic calibration is what makes them correctly abstain.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make smoke            # ~30s, synthetic, fail-fast regression check
```

The repo already ships the real data under `data/raw/` (12 monthly BTS-2024
parquets + 5 EUROCONTROL R&D Archive parquets + FAA registry + weather cache),
so the full pipeline runs offline with no downloads.

## Reproducing the results

```bash
make setup            # pip install -r requirements.txt
make smoke            # ~30s synthetic sanity check
make train            # NB00→NB03: builds the 150k sample, tunes the ladder
make evaluate         # NB04/05/06: thresholds, interpretation, EU transfer
make report           # NB99 end-to-end + HTML render
```

`make train` honours a configurable tuning budget; the fast default used for
submission is `RF_RANDOM_N_ITER=6 XGB_BAYES_N_ITER=10` (minutes on the 150 k
sample, not hours). Override on the command line for a heavier search.

Or run the notebooks in order: `00 → 01 → 02 → 03 → 04 → 05 → 06 → 99`.
**Notebook 00 is self-contained**: it builds the full 2024 frame
(`data/processed/flights.full2024.parquet`, provenance) and writes the
documented, seeded, month×label-stratified **150 k sample** as the canonical
modelling input (`data/processed/flights.parquet`). No hidden manual step.
`scripts/make_sample.py` rebuilds that sample standalone and uses the exact same
`src.data.sampling.stratified_modelling_sample` function.

Submission deliverables are pre-built in `reports/`:
`final_report.pdf`, `slides.pdf` (10 slides), `poster.pdf` (A1) — regenerate
the deck/poster with `python scripts/build_deliverables.py`.

## Repository layout

```
src/
  config.py                 # Paths, EC261 params, seeds
  data/
    loaders.py              # BTS / EUROCONTROL / FAA / weather loaders + augmenters
    bts_schema.py           # BTS CSV → internal schema mapping
    ec261.py                # Compensation tiers + EC261-eligible label
    sampling.py             # Seeded month×label-stratified modelling sample
    synthetic.py            # Deterministic synthetic generators (offline/CI)
  features/                 # Booking-time, historical (train-fold-only), cyclical
  models/                   # Model ladder factory, tuning spaces, calibration
  pipeline/                 # sklearn Pipeline assembly + temporal splits
  eval/                     # Profit metric, τ*, bootstrap, calibration, SHAP, failure modes
scripts/
  breakeven_analysis.py     # Root-cause structural-negative diagnostic
  make_sample.py            # Standalone rebuild of the documented sample
  build_deliverables.py     # Renders slides.pdf + poster.pdf
  smoke_test.py             # ~30s synthetic end-to-end (CI)
notebooks/                  # 00–06 + 99_final_pipeline (run top-to-bottom)
  archive/                  # Superseded notebook snapshots (not graded)
tests/                      # pytest suite (pipeline, label, metric, threshold, …)
reports/                    # final_report.{md,pdf}, slides.pdf, poster.pdf, figures/
data/raw/                   # Real BTS-2024, EUROCONTROL, FAA, weather (gitignored)
```

## Data sources

| Source | Role | Form on disk |
|---|---|---|
| **BTS Reporting-Carrier On-Time + Cause of Delay (2024)** | Primary training data — real, per-flight | `data/raw/bts_2024_01..12.parquet` (6.97 M flights) |
| **EUROCONTROL R&D Data Archive** | EU transfer validation (real, login-free) | `data/raw/eurocontrol_2023_{03,06,09,12}.parquet` + `2024_03` (3.89 M flights) |
| **FAA Aircraft Registry** | Aircraft age/type via tail-number join | `data/raw/faa_registry.csv` (~6 % null after join) |
| **Open-Meteo 24h forecast cache** | Booking-time weather features | `data/raw/weather_us_2024.parquet` |

Only single-year (2024) BTS is used, so the temporal split degrades to a
date-ordered 65/15/20 quantile split — stated as a limitation in the report,
not disguised. EUROCONTROL `FILED ARRIVAL TIME` already absorbs ATFM slot
delays, so transfer ranking is evaluated at the CODA-aligned ≥15 min threshold.

## Headline result (real 2024 BTS, test n = 29,604)

Every calibrated model converges on the same EV-optimal decision — **abstain**:
ROI 0 %, 0 buys. Uncalibrated models bet and lose €0.3–1.3 M. No global
threshold, no per-flight τ\*, no claim-success rate α ∈ [0.30, 0.95], and no
bankroll size is profitable. See `reports/final_report.pdf` for the full
analysis and `artefacts/breakeven_analysis.txt` for the structural proof.

## Team

Habib Rahal · Issam Arida · Adam Khoury · Lama Moucattash · Salma Nsour ·
Sanad ALbilleh. Every member commits their own work to GitHub; the syllabus
grades individual contribution via commit history.

## References

- Regulation (EC) No 261/2004, as amended September 2025; ECJ *Sturgeon* (C-402/07).
- BTS Reporting-Carrier On-Time documentation; EUROCONTROL R&D Data Archive + CODA reports.
- Pedregosa et al. (2011) scikit-learn; Chen & Guestrin (2016) XGBoost; Lundberg & Lee (2017) SHAP.
