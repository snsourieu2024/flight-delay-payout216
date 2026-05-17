# Presentation Speaking Script — 10 slides · 20 min + 5 min Q&A

Matches `reports/slides.pdf` exactly (flight-themed deck). Slides carry the
structure and visuals; **this script carries the 20 minutes of talking**.
Rehearse to the timing — the brief deducts marks for overrunning. Every member
speaks.

| # | Slide | Speaker | Target time |
|---|---|---|---|
| 1 | Title / boarding pass | Habib | 1:00 |
| 2 | Problem framing & why it is an ML problem | Issam | 2:15 |
| 3 | Step 1 · Data sourcing & the label decision | Issam | 2:15 |
| 4 | Step 2 · EDA & preprocessing decisions | Lama | 2:15 |
| 5 | Step 3 · Pipeline architecture & leakage control | Lama | 2:15 |
| 6 | Step 4 · Model ladder, calibration & objective | Adam | 2:15 |
| 7 | Step 5 · Evaluation methodology & rigour | Habib | 2:15 |
| 8 | The finding & the structural diagnosis | Adam | 2:30 |
| 9 | Step 6 · Transfer validation & interpretation | Sanad + Salma | 2:15 |
| 10 | Reflection, ethics & for the professor | Sanad | 1:25 |
| — | **Q&A** | all | 5:00 |

Total ≈ 20:00 + 5:00 Q&A.

---

## Slide 1 — Title (Habib, 1:00)
Welcome the room. One sentence each on what the project is and the punchline:
"We asked whether machine learning can find cheap flights whose EC261 payout
beats the ticket price. We engineered the full pipeline to test it honestly —
and we will show you, from two independent datasets, exactly why the answer is
no, and why that is a strong scientific result, not a failure." Introduce the
six team members. Set expectations: this is told as an ML-engineering
workflow, in the order we actually worked.

## Slide 2 — Problem framing (Issam, 2:15)
Explain EC261 mechanics: €250–600 for 3h+ arrival delays, but only if the
delay is carrier-attributable; weather, ATC and strikes are exempt. The key
intellectual move: we did not build "is it delayed?" — we built a *decision*:
buy a ticket iff predicted payout exceeds price plus claim and travel cost.
That reframes it as a ranking/EV problem, not an accuracy problem. State
plainly *why ML is even the right tool*: only if a learnable signal separates
payable from non-payable cheap flights — and testing that honestly, including
the possibility that no usable signal exists, is the whole project. Land the
"for the professor" point: we are graded on rigour and reasoning, and the
brief explicitly says a diagnosed negative result is worth as much as a
high-F1 model.

## Slide 3 — Data sourcing & label (Issam, 2:15)
Real data, not synthetic: 6.97 M US BTS 2024 flights joined with Cause-of-
Delay; plus 3.89 M EUROCONTROL flights reserved for transfer. The single most
consequential decision: the EC261-eligible label — 3h+ delay AND a
carrier/late-aircraft dominant cause. Explain the consequence: base rate drops
1.43% → 1.18%; modelling raw delay would have over-counted payouts and broken
the economics. Then the first real struggle, told as a story: 6.97 M rows made
tuning seven models intractable on a laptop; we did not just truncate — we
wrote a seeded, month×label-stratified 150 k sampler that preserves the base
rate (1.175%) and is reproducible.

## Slide 4 — EDA & preprocessing (Lama, 2:15)
Walk the decisions, not just the plots. Imbalance ~1% → accuracy is
meaningless (a constant "skip" scores 98.8%), so we rejected it and chose
PR-AUC, ROC-AUC, Brier, ECE and a money ROI. Missingness: aircraft age
recovered via the FAA join; the distance outlier was deliberately retained,
not winsorised, because EC261 pays on a distance step function — clipping
would corrupt the label. The struggle: weather features were silently 100%
null; we root-caused an un-wired loader call and now weather joins for 88.7%
of rows and is actually used. Note the weak feature–target correlation as the
first hint of what was coming.

## Slide 5 — Pipeline & leakage (Lama, 2:15)
This is the ML-engineering core. One serializable scikit-learn Pipeline:
14-day booking-horizon whitelist, forbidden columns dropped at step one,
train-fold-only rolling encoders, an expanding-window time-series split so no
future leaks into the past, and ColumnTransformer remainder='drop' as defence
in depth. The struggle: three notebooks had never been run end to end; we
enforced top-to-bottom execution and found and fixed three "zero-grade" bugs
before they could cost the whole grade. For the professor: the entire pipeline
runs top-to-bottom and is reproducible from a seeded script — please run
`make`.

## Slide 6 — Models, calibration & objective (Adam, 2:15)
Justify the ladder: trivial baselines to anchor no-skill, Logistic Regression
for interpretability, a tree, then Random Forest, XGBoost and an MLP. Every
model is isotonic-calibrated because the expected-value maths needs honest
probabilities. Explain the second-layer idea: the per-flight optimal threshold
τ*(T,d) derived in closed form, and that we tuned a *custom profit scorer*,
not F1 or AUC, because the objective is money. Note tuning was matched to each
model (Grid / Randomized / Bayesian) with a justified budget.

## Slide 7 — Evaluation rigour (Habib, 2:15)
Emphasise methodology. Metrics matched to an imbalanced decision task;
accuracy explicitly rejected and reported as such. Comparison on *both*
validation and test sets; expanding time-series cross-validation with the
reference model PR-AUC reported as 0.031 ± 0.014 (95% CI). ROC, learning and
confusion curves all produced. The struggle: Logistic Regression overflowed in
the learning-curve computation; we root-caused it and swapped to a tree-based
estimator — fixing the cause, not silencing the warning. The reliability
diagram shows why calibration is decisive.

## Slide 8 — The finding & structural diagnosis (Adam, 2:30)
The headline. Every calibrated model converges on the same decision: abstain —
test ROI 0%, zero buys; uncalibrated models actively bet and lose €0.3–1.0 M.
Tell the struggle honestly: this looked like a bug. We did not hand-wave — we
derived the closed-form break-even and proved it is structural EC261
arithmetic. Use the bar chart: required confidence ≈ 63%, the best
route×carrier cohort is only 7.7%, the base rate 1.2% — an ~8× gap no model
skill can close. For the professor: the negative result *is* the finding,
diagnosed, not a failure to optimise.

## Slide 9 — Transfer & interpretation (Sanad + Salma, 2:15)
*Sanad:* independent confirmation — scoring the BTS-trained model on 3.89 M
real EUROCONTROL flights, the ranking is perfectly inverted (Spearman
ρ = −1.00, top-k lift ≤ 1.03). Real EU structure exists (15.6%→39.2% by haul)
but the censored filed-plan label and domain shift mean the model cannot
exploit it.
*Salma:* interpretation — permutation importance plus a model-agnostic kernel
SHAP agree on schedule/route drivers; the forbidden cause-columns score
exactly 0.000, which empirically proves the leakage guard works.

## Slide 10 — Reflection, ethics & for the professor (Sanad, 1:25)
Close on rigour. Limitations stated honestly: synthetic fares are the largest
external-validity threat, but the conclusion is bounded by price-robust
break-even arithmetic, not just the model; single-year split; EUROCONTROL
filed-plan censoring. Ethics: speculating on others' disruption is legal but
marginal; cheap-fare selection bias; a US-trained model is not
EU-representative. What we would change: real fare data and a two-stage label.
Final line: ML correctly shows the strategy is structurally unprofitable,
proven from two independent sources — and all six of us contributed and are
presenting.

## Q&A preparation (all, 5:00)
1. *Dynamic pricing?* — Synthetic fares are the load-bearing caveat; the ~8×
   gap is far too large for realistic pricing to flip it.
2. *Cancellations?* — A separate EC261 article, deliberately scoped out.
3. *Why isotonic calibration?* — Non-parametric, strong on tabular data; and
   it is precisely what makes the models correctly abstain.
4. *Train on US, test on EU — is that valid?* — Presented as a transfer
   stress-test with the censored-label confound disclosed; ρ = −1.00 is the
   finding, not a defect.
5. *Biggest risk to the conclusion?* — Real fares; we bound it with the
   break-even arithmetic, not only the model output.
6. *Could a deeper model help?* — The MLP is included; no model family helps
   when the economically correct action is to abstain.
7. *What was the hardest engineering problem?* — Diagnosing the zero-ROI
   result as structural rather than a bug, via the closed-form break-even.
