# Poster Outline (A1 landscape, 841 × 594 mm)

The poster MUST stop the reader at three feet. The single most powerful asset
is the **per-flight threshold heatmap** — it explains the entire project in
one image. Use it as the largest panel.

## Layout (3 rows × 4 columns)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TITLE: "Profitable Flight Delays Under EC261"                              │
│  Authors  •  Affiliation: IE University  •  Course code                     │
├──────────────────┬─────────────────────────────────┬────────────────────────┤
│ MOTIVATION       │       PER-FLIGHT THRESHOLD       │ DATASET               │
│ - EC261 pays     │       HEATMAP τ*(T, d)           │ - BTS 2018-2024       │
│   €250-€600      │       (the headline figure)       │ - 42M flights        │
│ - Cheap ticket   │                                   │ - FAA + NOAA joins   │
│   + likely delay │                                   │ - EUROCONTROL ADRR   │
│   = arbitrage?   │                                   │   (EU validation)    │
├──────────────────┼─────────────────────────────────┴────────────────────────┤
│ METHODS          │  RESULTS PANEL                                            │
│ Pipeline:        │  ┌─────────────┬──────────────┬──────────────────────┐ │
│ - leakage audit  │  │ Model       │ ROC-AUC      │ ROI per-flight       │ │
│ - booking-time   │  │ Dummy       │ 0.500        │ n/a                  │ │
│   features       │  │ LogReg      │ 0.xx         │ x.x%                 │ │
│ - 6-model ladder │  │ RandomForest│ 0.xx         │ x.x%                 │ │
│ - isotonic cal   │  │ XGBoost+iso │ **0.xx**     │ **x.x%**             │ │
│ - Bayesian tune  │  └─────────────┴──────────────┴──────────────────────┘ │
│ - SHAP           │  Profit-vs-threshold curve (sparkline)                    │
│                  │  Bankroll-constrained ROI by budget (small bar)           │
├──────────────────┼─────────────────────────────────┬────────────────────────┤
│ INTERPRETATION   │ FAILURE MODES                    │ REFERENCES & TOOLS    │
│ Top SHAP feats:  │ - evening hub departures         │ EC261, BTS, FAA       │
│ - route 90D rate │   in winter overconfident        │ scikit-learn, XGBoost │
│ - dep hour       │ - weather causes look like       │ SHAP, scikit-optimize │
│ - tail 365D rate │   carrier delays in features     │ EUROCONTROL ADRR       │
│ - aircraft age   │ - EU transfer: monotonic deciles │ GitHub: <repo URL>    │
└──────────────────┴─────────────────────────────────┴────────────────────────┘
```

## One-paragraph elevator (top of poster)

> Flight-delay prediction usually maximises F1 or ROC-AUC. We argue that under
> EC261 — which pays €250-€600 for delays of 3+ hours that are not weather or
> ATC-attributable — the right metric is **expected ROI on a bankroll-
> constrained portfolio**. The optimal threshold is *flight-specific*: a €30
> short-haul ticket needs a higher predicted P(delay) than a €300 long-haul
> ticket. Calibration is not optional. We trained six models on the BTS dataset
> with a leakage-audited pipeline and validated transfer to EUROCONTROL ADRR.

## Visual hierarchy guidelines

1. The τ*(T, d) heatmap is **the** poster.  ~25% of total area.
2. Single results table, three columns max, ranked by ROI.
3. One sparkline-style profit-vs-threshold plot.
4. SHAP bar plot for top-10 features.
5. A simple confusion-matrix-style failure-mode 2×2.
6. Avoid walls of text. Use ≤2 sentences per panel.

## Build instructions

- LaTeX `beamerposter` template OR Figma A1 template (instructor permits both).
- Colour palette: matplotlib default `tab10`, no rainbow. Reserve red for losses.
- Embedded fonts. PDF must be printer-ready at 300 DPI.
- Department offers printing — submit by **20 May** deadline.
