# Poster Outline (A1 landscape, 841 × 594 mm)

Single message: **a rigorously diagnosed negative result**. The hero asset is
the **break-even gap** — required confidence vs achievable probability.

## Layout (title band + 3 rows × 3 columns)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Profitable Flight-Delay Prediction Under EC261 — A Diagnosed Negative      │
│ Result   •   6 authors   •   IE University   •   BCSAI2025CSAI.2.M.A C2    │
├──────────────────┬───────────────────────────────┬────────────────────────┤
│ MOTIVATION       │  HERO: BREAK-EVEN GAP          │ DATA                   │
│ EC261 €250–600   │  required τ* ≈ 0.63 vs best   │ Real BTS 2024 6.97 M → │
│ for carrier 3h+  │  cohort 7.7 % vs base 1.2 %   │ seeded 150 k sample    │
│ delays. Cheap    │  → ~8× structural gap         │ EC261 base rate 1.18 % │
│ ticket + likely  │  (τ*(T,d) heatmap +           │ EUROCONTROL 3.89 M     │
│ delay = +EV?     │   profit-curve panel)         │ (transfer test)        │
├──────────────────┼───────────────────────────────┼────────────────────────┤
│ METHOD           │  RESULTS (real test, n=29,604)│ EU TRANSFER            │
│ leakage-audited  │  Model(+iso) ROC-AUC PR ROI   │ Spearman ρ = −1.00     │
│ pipeline; 6-model│  Dummy  0.500 .0081 0.0%      │ (ranking inverted);    │
│ ladder; isotonic │  LogReg 0.557 .0100 0.0%      │ top-k lift ≤ 1.03;     │
│ calibration;     │  RF     0.576 .0102 0.0%      │ EU has real signal     │
│ Grid/Rand/Bayes  │  XGB    0.587 .0107 0.0%      │ (15.6%→39.2% by haul)  │
│ on a profit      │  → ALL abstain: 0 buys,       │ the BTS model cannot   │
│ scorer           │  −€19,941. Uncal XGB −187.6 % │ exploit                │
├──────────────────┼───────────────────────────────┼────────────────────────┤
│ INTERPRETATION   │ WHY IT FAILS (structural)     │ REFLECTION / REFS      │
│ Perm. importance:│ EC261 cap €250–600 × α0.65    │ Negative result =      │
│ FL_DATE, dep-hr, │ < ticket+€65 friction for a   │ finding (brief §1/§2). │
│ aircraft age,    │ ~1 % event. No α, no τ, no    │ Limits: synthetic      │
│ distance.        │ bankroll is profitable.       │ fares, 1-yr split.     │
│ Cause cols 0.000 │ Calibration → correct         │ scikit-learn, XGBoost  │
│ (leakage guard). │ abstention.                   │ GitHub: <repo>         │
└──────────────────┴───────────────────────────────┴────────────────────────┘
```

## Elevator (top strip)

> Flight-delay prediction usually maximises F1. Under EC261 the right objective
> is expected ROI on a bankroll. We built a leakage-audited 6-model pipeline
> with isotonic calibration and a custom profit metric on real 2024 BTS data.
> Result: **no positive-EV ticket exists** — every calibrated model correctly
> abstains (ROI 0 %). A closed-form break-even shows the required confidence
> (~63 %) is ~8× any achievable delay rate; a 3.89 M-flight EUROCONTROL
> transfer test independently confirms it (ρ = −1.00). This is the finding.

## Panels → figures (reuse real PNGs from reports/figures/)

1. Hero left: `04_tau_heatmap.png` (required τ*) beside `04_profit_curve.png` (no profitable point)
2. Calibration: `05_calibration_before_after.png`
3. Interpretation: `05_permutation_importance.png`
4. EU: `06_decile_monotonicity.png` (inverted) + `06_tier_concentration.png`
5. Confusion: `04_confusion_at_tau.png` (zero positive predictions = correct abstention)

## Build
- A1 landscape 841×594 mm, 300 DPI, embedded fonts, printer-ready PDF.
- Palette: muted; red reserved for losses. ≤2 sentences per panel.
- Department prints; deadline **20 May**.
