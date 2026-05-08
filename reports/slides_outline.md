# Slide Deck Outline (10 slides, 20 min + 5 min Q&A)

Speaker rotation: every group member speaks. ~2 minutes per slide.

---

## Slide 1 — Title (1 min)
**Profitable Flight Delays Under EC261: A Decision-Theoretic Approach**
- Team names, IE University, course code
- One-line thesis: "We treat flight-delay prediction as a positive-EV betting problem, not a classification problem."

## Slide 2 — Why this is interesting (1.5 min)
- EC261 pays €250–€600 for 3+ hour delays — but only if airline-attributable
- The September 2025 amendment tightened the extraordinary-circumstances list (weather, ATC, ANSP strikes excluded)
- Question: can a buyer with a fixed bankroll find positive-EV cheap tickets?
- Lead with the *τ\*(T, d) heatmap* as the visual hook

## Slide 3 — Data and labelling (2 min)
- BTS 2018-2024 (~42M flights) as primary substrate
- EUROCONTROL ADRR as EU transfer-validation case study
- The labelling decision: EC261-eligible only when cause ∈ {Carrier, Late Aircraft}
- Show the funnel: 100% delays → ~25% are 3h+ → ~60-70% of those are eligible

## Slide 4 — Pipeline and leakage audit (2 min)
- Pipeline diagram (raw → BookingTimeFeatureBuilder → HistoricalDelayRateEncoder × 4 → ColumnTransformer → classifier)
- The audit table: every BTS column classified as allowed or forbidden
- TimeSeriesSplit, not random KFold

## Slide 5 — Model ladder and results (2.5 min)
- Trivial → Classical → Advanced
- Results table sorted by ROI (the headline metric)
- Calibration matters: Brier and ECE before/after isotonic
- One reliability-diagram inset

## Slide 6 — The "second layer" (2.5 min)
- Per-flight threshold τ\*(T, d) — the formula and the heatmap
- Profit-vs-global-threshold curve
- Bankroll-constrained policy results at €1k, €10k, €100k
- Bootstrapped 95% CI on ROI

## Slide 7 — Sensitivity analysis (1.5 min)
- Vary α (claim-success rate): when does strategy turn unprofitable?
- Vary travel cost
- Vary the EC261 distance tiers
- Speak to the limits of the empirical robustness

## Slide 8 — SHAP and failure modes (2 min)
- Top-10 SHAP features
- One local-explanation case study (a marginal flight at τ\*)
- Failure-mode bins by hour and origin
- Most expensive false positives are evening hubs in winter — weather-driven delays look like carrier delays in our features

## Slide 9 — EU transfer validation (2 min)
- Decile monotonicity plot from notebook 06
- Top-1% lift on EUROCONTROL ADRR sample
- What transfers, what doesn't, and why

## Slide 10 — Reflection and limitations (3 min)
- What worked: per-flight threshold + isotonic calibration
- What didn't: synthetic ticket prices are a load-bearing assumption that real fare data could break
- What surprised us: the EC261-aware label removed ~30-40% of "positives" — relabelling was the highest-leverage decision
- What we would do differently: two-stage architecture (delay prediction + carrier-attributability prediction)
- One-sentence honest takeaway: *"The strategy is positive-EV in our model but its viability in production rests on real fare data we did not have."*

## Q&A prep — likely questions

1. *"Have you accounted for dynamic pricing?"* — No, that is the load-bearing limitation; address it head-on.
2. *"What about cancellations?"* — Different EC261 article, deliberately scoped out.
3. *"Why isotonic over Platt?"* — Isotonic is non-parametric and almost always wins on tabular data.
4. *"Why train on US data and predict on EU?"* — Because EU per-flight cause-coded data is embargoed two years and we needed seven years of training data.
5. *"What's the biggest risk to this strategy?"* — That airlines deny the claim. We model claim success at α=0.65 and sensitivity-analyse around it.
6. *"Could you have used a deep model?"* — Tabular structure plus modest sample size means tree ensembles are the right family. We included an MLP for completeness; it did not win.
