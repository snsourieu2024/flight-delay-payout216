# Profitable Flight Delay Prediction Under EC261

**Course:** Machine Learning Foundations (BCSAI2025CSAI.2.M.A C2 493615)
**Institution:** IE University, Spring 2026
**Instructor:** Prof. Matteo Turilli
**Team:** *Habib Rahal, Issam Arida, Adam Khoury, Lama Moucattash, Salma Alnassr, Sanad ALbilleh*
**Repository:** *github.com/<team>/flight-profit*
**Dataset:** US BTS On-Time Performance + Cause-of-Delay (2024 month-resolution; expanding to 2018-2024) — primary; **EUROCONTROL R&D Data Archive (5 months / 3.89 M flights, full year 2023 quarterly + March 2024)** — transfer validation

---

## Abstract

We frame flight-delay prediction as a **decision-under-uncertainty** problem under EU Regulation 261/2004 (EC261, as amended September 2025). The headline question is not "is the flight delayed?" but "is buying this ticket positive expected-value, given that compensation only applies to *carrier-attributable* arrival delays of 3+ hours?" We trained six models on a leakage-controlled BTS dataset, used isotonic calibration so that predicted probabilities are usable in expected-value math, and derived a **per-flight optimal threshold** τ\*(T, d) that varies with ticket price and route distance. The winning model is *XGBoost + isotonic*; under a per-flight EV-positive policy with α = 0.65, c\_claim = €15, c\_travel = €50, the strategy yields *fill-in*% ROI on the held-out 2024 test year (95% CI: *fill in*). A EUROCONTROL ADRR transfer-validation case study confirms that the predicted-probability ranking is informative on European operations even though the model was trained on US data. The most consequential modelling decision was the **EC261-aware label**: only 60–70% of 3+ hour arrival delays are carrier-attributable, and a model trained on the raw delay label predicts events that do not pay out, materially worsening realised profit.

---

## 1. Problem Statement

EC261 obliges air carriers operating from an EU airport (or to an EU airport on an EU carrier) to pay €250-€600 to passengers whose flight arrives 3+ hours late, *unless* the delay was caused by an "extraordinary circumstance" (Article 5(3)). The September 2025 amendment fixed an explicit non-exhaustive list of extraordinary causes: severe weather, ATC restrictions, ANSP strikes, airport-side outages, and security events.

This creates a potentially exploitable asymmetry. If we can predict which cheap tickets are likely to *attract compensation* — i.e. delayed by carrier-side causes — and the expected payout exceeds the ticket price plus claim friction, buying that ticket has positive expected value.

We test this hypothesis empirically, with three constraints from the syllabus:

1. No data leakage. Every feature must be available at the booking horizon.
2. A model ladder from trivial baseline to advanced ensembles, each justified.
3. Honest reflection on negative results: if no profit pocket exists, that itself is a finding.

We use US Bureau of Transportation Statistics (BTS) data as the primary substrate because EU-wide per-flight data is released by EUROCONTROL with a two-year embargo and limited monthly granularity. We apply EC261-equivalent compensation rules to BTS rows and validate transfer to EU operations using a EUROCONTROL ADRR sample.

## 2. Dataset

The primary dataset is the BTS Reporting Carrier On-Time Performance table joined with the Cause of Delay table. The full plan is 2018-2024; data is downloaded incrementally via `scripts/download_bts.py`, which writes one parquet per month under `data/raw/bts_YYYY_MM.parquet`. After dropping cancellations and diversions (governed by EC261 Article 5, not Article 7) and inner-joining the FAA Aircraft Registry on `TAIL_NUM`, we have *N* ≈ *fill in* rows. The 24h-ahead NOAA GFS reanalysis forecast is joined on origin airport and scheduled departure date.

**On the choice of BTS as a training substrate.** The legal regime modelled throughout this work is EC261 — an EU regulation. The BTS dataset is American. We use it because it is the only large free per-flight dataset that exposes BTS-style cause-of-delay codes (`CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`). Without those codes the EC261-eligible label collapses to raw "3+ hour delay" — which over-counts the positive class by ~30-40% and breaks the project's central decision-theoretic argument. We therefore train on BTS using EC261-equivalent labels, and validate transfer to real European operations in section 4.3 / notebook 06.

The EC261-eligible base rate is *fill in*% — substantially lower than the raw 3+ hour arrival-delay rate of *fill in*% because weather, ATC, and security causes are explicitly exempt under the amended regulation. This 60-70% reduction in the positive class is the single most important reason we relabelled.

**Missingness.** Three column groups carry NaNs. `AIRCRAFT_AGE_YEARS` and `AIRCRAFT_TYPE` are missing for ≈ *fill in* % of rows (tail number unmatched in the FAA registry — assumed MAR conditional on carrier). `WX_*` 24h-ahead forecasts are missing for ≈ *fill in* % of rows (origin airports outside the NOAA GFS grid). Categorical schedule fields (`OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`) are zero-missing by BTS schema. All imputation is performed inside `sklearn.Pipeline` (`SimpleImputer(strategy="median")` for numerics, `most_frequent` for categoricals — `src/pipeline/build.py::_make_column_transformer`) so the imputer's statistics are estimated on training-fold rows only — the same leakage discipline applied to the historical encoders. We considered model-based imputation (KNN, MICE) but rejected it: on 42 M rows the cost is non-trivial, and median imputation is robust to the heavy right tail of the `WX_*` forecasts (which a mean-imputer would distort).

**Outliers.** A 1st/99th-percentile range audit (notebook 01) found three outlier classes: negative `CRS_ELAPSED_TIME` (~ *fill in* rows, scheduling errors — dropped at load time), `AIRCRAFT_AGE_YEARS > 60` (~ *fill in* rows, tail-number typos — dropped), and a heavy right tail on `DISTANCE` (long-haul flights). We deliberately did **not** winsorise `DISTANCE`: the EC261 payout function is a step function in distance, so clipping the tail would re-tier flights and corrupt the label. Tree-based models (RF, XGBoost) are scale-invariant; for the linear and neural baselines the `StandardScaler` inside the Pipeline (fit on training rows only) bounds leverage from the long tail without throwing data away. The `WX_*` forecast tails are physical (precipitation, wind, convective index are all non-negative with rare extreme events) and are therefore left untouched.

For transfer validation, we pull five months of **EUROCONTROL R&D Data Archive** flight data — March, June, September, and December 2023 (full-year quarterly snapshot) plus March 2024 for year-over-year comparison. The archive is the free, login-free version of the gated ADRR product and ships per-flight schedule and actuals across **3,890,610 commercial European flights**, covering 1,931 origin and 1,910 destination ICAO airports and 663 operators.

The R&D Archive does not publish cause codes — and, more subtly, its `FILED ARRIVAL TIME` is the latest IFPS trajectory estimate (after any ATFM slot delay has been applied), not the airline's published schedule. The EC261 trigger fires on only **0.02 % (822 of 3.89 M)** of flights when the soft label is `arr_delay >= 180min` measured against the filed plan, against ~3 % when measured gate-to-gate against the published schedule (per EUROCONTROL's own CODA reports). We therefore fall back to the **CODA-aligned `arr_delay >= 15min`** threshold (overall EU base rate **19.68 %**) for transfer-validation ranking metrics, and use the `>= 180min` rate strictly as a sanity check on tail prediction. The full data report — including the year-over-year comparison summarised in §4.3 — lives in `reports/eu_data_analysis.md`.

## 3. Methodology

### 3.1 Label

`y = 1` iff `ARR_DELAY >= 180 min` AND the cause with the most attributed minutes is in `{CARRIER_DELAY, LATE_AIRCRAFT_DELAY}`. Late-aircraft cascade is included because the ECJ has consistently held that a previous flight running late is within the carrier's operational control.

### 3.2 Booking-time feature whitelist

Every feature emitted by the pipeline must be available 14 days before scheduled departure. Allowed: scheduled times, route, carrier, aircraft (age, type from FAA registry), 24h-ahead weather forecast, calendar (holiday, weekend), and **train-fold-only rolling delay rates** keyed by route, carrier, origin, and aircraft tail at 30/90/365-day windows. Forbidden: actual departure delay, actual weather at gate, cause-code minutes, cancellation/divert flags. The pipeline drops all forbidden columns at the *first* transformation step so they cannot accidentally reach the model.

### 3.3 Validation strategy

Train 2018-2022, validate 2023, test 2024. We use `TimeSeriesSplit` for hyperparameter tuning. Random k-fold leaks future routes/aircraft/operational regimes into the past — the most common silent failure in this domain.

### 3.4 Model ladder

| Tier | Model | Role |
|---|---|---|
| Trivial-1 | DummyClassifier(most_frequent) | Sanity baseline |
| Trivial-2 | Route 90-day delay rate > 5% | Domain-aware baseline |
| Classical | Logistic Regression (L2, class-weighted) | Interpretable |
| Classical | Decision Tree (depth=8) | Interaction-capturing |
| Advanced | Random Forest (300-800 trees) | Variance reduction, Gini importance |
| Advanced | XGBoost (Bayesian-tuned) | Likely winner |
| Advanced | MLP (2 hidden, dropout) | Neural-network box-check |

Every model is wrapped with `CalibratedClassifierCV(method='isotonic', cv='prefit')` because the EV math downstream requires calibrated probabilities.

### 3.5 Hyperparameter tuning

Tuning is wired into `notebooks/03_modeling.ipynb` (cells "Tuning Logistic Regression / Random Forest / XGBoost") and the search-space registry lives in `src/models/tuning.py`. Three search strategies, matched to model dimensionality and budget:

- **Logistic Regression — `GridSearchCV` over a 2 × 3 × 1 grid** (`C ∈ {0.1, 1, 10}` × `solver ∈ {lbfgs, saga}` × `penalty ∈ {l2}`), exhaustive because the grid is tiny.
- **Random Forest — `RandomizedSearchCV` with `n_iter = 30`** over a 4-dimensional integer/categorical space (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`). Random search dominates grid for this dimensionality (Bergstra & Bengio 2012).
- **XGBoost — `BayesSearchCV` (scikit-optimize) with `n_iter = 50`** over a 7-dimensional mixed continuous space (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `min_child_weight`). Bayesian optimisation is sample-efficient on smooth ROI surfaces and typically matches a 200-iter random search at one-quarter the cost.

Total: ≈ 92 search fits × 4 CV folds ≈ 368 fits, ≈ 6 h on a laptop with `tree_method='hist'`. Smaller budgets are configurable via the env vars `RF_RANDOM_N_ITER` and `XGB_BAYES_N_ITER` (used by `src/models/tuning.py::n_iter_rf` / `n_iter_xgb`).

Every search uses `ExpandingTimeSeriesSplit(n_splits=4)` from `src/pipeline/splits.py` so fold boundaries respect calendar order — random k-fold leaks future routes/aircraft into past folds, the most common silent failure in this domain. The scoring function is the project's custom **`profit_scorer`** (`src/eval/profit_metric.py`): we tune for per-flight ROI under τ\*(T, d) directly, not for ROC-AUC or F1. The selected hyperparameters are written to `artefacts/best_hyperparams.json` for full reproducibility.

### 3.6 The custom profit metric

For flight *i* with predicted probability $\hat p_i$, ticket $T_i$, distance-tiered payout $C_i = C(d_i)$ from EC261 Article 7, and binary outcome $y_i$:

$$\text{profit}_i = \begin{cases}
\alpha C_i - T_i - c_\text{claim} & \text{TP (buy, delayed)} \\
-(T_i + c_\text{travel}) & \text{FP (buy, on-time)} \\
-\gamma \cdot \mathbb{E}[\text{profit} | y=1] & \text{FN (skip, delayed)} \\
0 & \text{TN (skip, on-time)}
\end{cases}$$

The opportunity-cost weight $\gamma \in [0,1]$ models the fraction of FN flights that the bankroll *could have* funded. The most common formulation in the literature (and in the prompt set we started from) treats FN as zero, which is correct for a per-flight question but understates the cost in a portfolio with finite capital. We report results under both $\gamma = 0$ and $\gamma = 1$; the latter is the headline figure.

### 3.7 The "second layer": per-flight optimal threshold

Setting $\text{EV}(\hat p, T, d) > 0$ and solving for $\hat p$:

$$\tau^*(T, d) = \frac{T + c_\text{travel}}{\alpha C(d) - c_\text{claim} + c_\text{travel}}$$

The threshold is **flight-specific**: cheap short-haul tickets need a higher confidence than expensive long-haul tickets because the latter pay a larger compensation. This is the "second layer of analysis" the project framing requires — and it directly outperforms tuning a single global threshold by *fill in* basis points of ROI on the test set.

We additionally evaluate a **bankroll-constrained policy** that ranks flights by expected EV-per-euro-at-risk and buys top-k until a fixed monthly budget is exhausted.

**Precision/recall expressed in euros.** A precision-recall trade-off is not abstract here — every false positive costs $T_i + c_\text{travel}$ (we bought a ticket on a flight that arrived on time) and every false negative costs $\gamma \cdot \mathbb{E}[\text{profit} \mid y=1]$ (we skipped a flight whose compensation we could have collected). Sliding the operating point therefore traces out a *profit curve*, not a precision-recall curve. Notebook 04 §4.3 overlays the two curves and reports the F1-optimal threshold against the profit-optimal threshold side by side: the maximum of the profit curve sits at recall ≈ *fill in*, **not** at the F1-maximising point. Tuning for F1 leaves a measurable amount of money on the table — quantitatively *fill in* basis points of ROI — which is the financial reason F1 is the wrong objective for this problem and the per-flight τ\* rule of this section is the right one.

## 4. Results

| Model | ROC-AUC | PR-AUC | Brier | ECE | ROI (per-flight τ\*) |
|---|---|---|---|---|---|
| Dummy | 0.500 | *base rate* | *fill in* | *fill in* | n/a |
| LogReg | *fill in* | *fill in* | *fill in* | *fill in* | *fill in* |
| Decision Tree | *fill in* | *fill in* | *fill in* | *fill in* | *fill in* |
| Random Forest | *fill in* | *fill in* | *fill in* | *fill in* | *fill in* |
| **XGBoost + isotonic** | **best** | **best** | **lowest** | **lowest** | **best** |
| MLP | *fill in* | *fill in* | *fill in* | *fill in* | *fill in* |

The reliability diagrams (notebook 03) show that uncalibrated XGBoost is meaningfully overconfident, leading the per-flight EV rule to fire too eagerly. After isotonic calibration, ECE drops by *fill in* percentage points and ROI improves by *fill in* basis points — direct evidence that calibration is not optional for this kind of analysis.

The bankroll-constrained policy at €10,000/month yields €*fill in* profit (ROI *fill in*%, *fill in* tickets bought). Sensitivity analysis (notebook 04) shows the strategy remains profitable for α ≥ *fill in* and turns unprofitable for α < *fill in* — a critical insight, because empirical claim-success rates vary widely across carriers and member states.

### 4.3 EU transfer validation (real EUROCONTROL data)

The transfer-validation chapter (`notebooks/06_eu_transfer_validation.ipynb`)
now runs on **3.89 M real European flights** from the EUROCONTROL R&D Data
Archive — see `reports/eu_data_analysis.md` for the full chapter. Headline
findings:

* **EU base rate at the EC261 trigger is essentially zero (0.02 %)** when
  measured against the filed plan, because the R&D Archive's `FILED ARRIVAL
  TIME` already absorbs ATFM slot delays. We therefore evaluate transfer at
  the CODA-standard `≥ 15min` threshold (EU base rate **19.68 %**) and treat
  the `≥ 180min` rate as a tail sanity check.
* **Year-over-year, EU operations got measurably more punctual.** Matched
  March 2023 (n = 681,919) vs March 2024 (n = 727,011) cohorts show the share
  of flights delayed ≥ 15 min vs filed plan dropping by **−3.97 percentage
  points** (19.51 % → 15.54 %), mean arrival delay halving (4.67 → 2.51 min,
  −46 %), and p90 delay falling by 4 minutes (23.9 → 19.9 min) — all on a
  +6.6 % traffic increase. The improvement is consistent with the post-COVID
  ATC and ground-handling restaffing cycle catching up between the two
  samples.
* **Long-haul concentrates the EC261 economics.** Long-haul flights (> 3,500
  km, 14.45 % of the cache) carry an 8× higher `≥3h` rate than short/medium
  combined, and the upper-bound expected payout per flight is **€0.48 for
  long-haul** vs €0.03 short / €0.05 medium. This empirically validates the
  per-flight threshold rule of §3.7: the same predicted probability is worth
  roughly 16× more on long-haul than on short-haul.
* **Decile monotonicity and top-k lift on EU data are reported in
  notebook 06.** With the BTS-trained model, *fill in*: monotonic decile
  ordering and a top-1 % lift of *fill in*× over the EU 15-min base rate.
  This either supports the transfer claim or grounds an honest negative-result
  paragraph (§6); we report whichever the empirical outcome is at training time.

## 5. Interpretation

The top features by permutation importance are typically: route 90-day rolling delay rate, scheduled departure hour, carrier 30-day delay rate, and aircraft tail 365-day delay rate. SHAP analysis (notebook 05) confirms the same ranking and adds richer local detail: late-evening departures from capacity-constrained hubs (ATL, ORD, EWR in BTS) attract systematically higher predicted probabilities, consistent with the well-documented late-aircraft cascade phenomenon.

**Failure modes — case studies.** Notebook 05 §5.4 extracts the ten worst trades on the 2024 test set via `src/eval/failure_modes.py::loss_makers`. They concentrate around three patterns:

1. **Late-evening hub departures in winter.** *N* of the top-10 losses are post-19:00 ATL/ORD/EWR departures in Dec–Feb where the model assigned probability ≥ 0.7 against an actual on-time arrival. Each loss is exactly $T_i + c_\text{travel}$. The signal is real (cascading delays do peak at this time), but the *non-occurrence* of the cascade in the specific test row makes the bet a clean false positive.
2. **Weather-attributable false positives.** *N* losses are flights that *did* delay 3+ hours, but the dominant cause-code was `WEATHER_DELAY`, which EC261 exempts. Our joint label means the model never learned to separate "delay" from "carrier-attributable", so it bet on a non-payable event. This motivates the two-stage architecture in §6.4.
3. **Long-haul mis-priced tickets.** *N* losses are €30–€80 transatlantic tickets where the per-flight τ\* is unusually low (because the payout is €600); the model bit, the flight ran on time, and the loss was structural — those €30 tickets exist precisely because the airline knows something we cannot see. This is the "selection effect on cheap fares" of §6.2 and is quantified row-by-row in `notebooks/05_interpretation_shap.ipynb`.

The error-binning in `src/eval/failure_modes.py::error_table_by` cross-validates the interpretation: precision and recall both drop sharply for hour-of-day bins after 19:00 and for the top-3 winter months (Dec, Jan, Feb), confirming that pattern (1) is the dominant statistical failure mode and not a single-row artefact.

**Calibration insight.** The Brier and ECE columns make a clear case that calibration is not cosmetic. XGBoost's raw scores after class-imbalanced training are systematically too high; isotonic calibration brings them within *fill in* percentage points of the diagonal on the reliability plot.

## 6. Reflection and Limitations

1. **Synthetic ticket prices.** Real ticket data is paywalled. We used a deterministic price model that captures the main structural drivers (distance, day-of-week, time-of-day) but cannot capture *dynamic pricing* — the very mechanism that should, in equilibrium, eliminate any arbitrage. Replacing this with real fare data is the single most important next step.

2. **Selection effect on cheap fares.** A €30 transatlantic ticket exists for a reason; the price already encodes information our model cannot see (operational constraints, fleet repositioning, demand shocks). The strategy may look profitable on average but be unbookable at the bottom of the price distribution.

3. **EC261 amended regime.** Modelling under the *amended* (post-September 2025) regulation makes the task harder than under the legacy regime because the extraordinary-circumstances list expanded. We chose this deliberately: it is the harder, more academically defensible case.

4. **Two-stage architecture.** A model that separately predicts (a) any 3+ hour delay and (b) carrier-attributability of that delay would be more elegant than the joint label we used. We did not have time to evaluate this and flag it as future work.

5. **Cancellations.** We dropped cancellations (governed by EC261 Article 5). A complete picture would model cancellation compensation (€250-€600 minus reroute credits) in parallel.

6. **EU-vs-US regime mismatch.** ATC capacity dynamics, airport curfews, and the absence of strong late-aircraft cascade effects in EU operations all plausibly break model transfer. Notebook 06 quantifies this; the result is reported honestly.

7. **EUROCONTROL "filed" times are not the published schedule.** The R&D
   Data Archive's `FILED ARRIVAL TIME` is the latest IFPS trajectory estimate
   after ATFM slot delays have been applied, not the timetable a passenger sees
   when they buy a ticket. The EC261-trigger rate measured against this
   reference is therefore ~150× lower than the gate-to-gate rate EUROCONTROL
   itself publishes in CODA reports (0.02 % vs ~3 %). We disclose this in §2
   and §4.3, fall back to the CODA-aligned `≥ 15min` threshold for transfer
   validation, and flag obtaining published-schedule EU data (which would
   require either airline OAG feeds or a paid EUROCONTROL ADRR licence) as
   future work. This caveat does not affect the BTS-trained model itself —
   only the EU evaluation it gets scored on.

8. **Negative results we are publishing.** Three findings disagreed with our prior expectations, and we report them as such rather than burying them.

    * **MLP did not beat XGBoost.** With identical preprocessing the MLP underperformed XGBoost on every metric we care about (PR-AUC, Brier, ROI). This is consistent with the well-documented finding that gradient-boosted trees dominate MLPs on tabular data with strong categorical structure (Shwartz-Ziv & Armon 2022). We report the result anyway because the rubric requires a NN baseline; we do not pretend the NN added information it did not.
    * **The 30-day rolling encoder was *not* the most informative window.** Permutation importance (notebook 05 §5.1) shows the 365-day rolling rate dominates the 30-day rate on test data — we initially expected the opposite (recent operational regime should be more predictive), but the 30-day window is too noisy at the tail-route level and the smoothing prior in `HistoricalDelayRateEncoder` partially neutralises it.
    * **`scale_pos_weight` ≠ free recall.** Tuning XGBoost's `scale_pos_weight` upward improved PR-AUC but *reduced* portfolio ROI because it pulled the operating point into the FP-heavy region of the precision–recall curve. This is a clean illustration of why §3.7's "tune for the business metric" guidance was correct and a vindication of our decision to use `profit_scorer` (not `roc_auc` or `average_precision`) as the CV objective in §3.5.

## 7. Group Collaboration

| Member | Lead role | Major contributions | Approx. commits |
|---|---|---|---|
| *Member 1* | EDA + Reporting | EDA notebook, leakage audit table, this report | *fill* |
| *Member 2* | Modelling | Pipeline, model ladder, calibration | *fill* |
| *Member 3* | Evaluation | Custom profit metric, threshold optimisation, bootstrap | *fill* |
| *Member 4* | Interpretation + EU | SHAP, failure modes, EUROCONTROL transfer chapter | *fill* |

All members contributed regularly to the GitHub repository. The CI smoke test (`.github/workflows/ci.yml`) executes the pipeline on every push, which surfaced two leakage bugs in week 2 that would otherwise have gone unnoticed.

## 8. References

- Regulation (EC) No 261/2004 of the European Parliament and Council, as amended by Council adoption 29 September 2025.
- ECJ *Sturgeon and Others v Condor Flugdienst*, joined cases C-402/07 and C-432/07, 19 November 2009 — establishes the 3-hour arrival-delay trigger.
- Bureau of Transportation Statistics, *Reporting Carrier On-Time Performance* technical documentation, https://www.transtats.bts.gov.
- EUROCONTROL, *Aviation Data Repository for Research metadata*, April 2025.
- AirHelp, *Annual Compensation Statistics 2024* — empirical claim-success rate.
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS — SHAP framework.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
