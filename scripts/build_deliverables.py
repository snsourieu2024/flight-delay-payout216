"""Build submission deliverables from real artifacts (no external tooling).

Outputs:
  reports/slides.pdf   -- 10-slide flight-themed deck (16:9), content-rich,
                          built for a 20-minute talk (script in
                          reports/slides_outline.md)
  reports/poster.pdf   -- single A1 landscape poster (841 x 594 mm)

Pure matplotlib so it renders deterministically. Layout uses an auto-stacking
column engine with text wrapping so dense content never overlaps. All numbers
are the real, weather-active values from the executed notebooks.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"

PAPER = "#f5f2ea"
NAVY = "#0a2540"
OCEAN = "#13507a"
SKY = "#3a8fb7"
MIST = "#dde9f1"
INK = "#11202b"
SAND = "#e8a13a"
ALERT = "#c0432e"
GO = "#2f7d62"
MUTE = "#6f7f88"
LINE = "#d6dee2"
BG, TEAL, AMBER, LOSS = PAPER, OCEAN, SAND, ALERT

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "savefig.facecolor": PAPER, "figure.facecolor": PAPER})

SPK = dict(habib="Habib", issam="Issam", lama="Lama", adam="Adam",
           salma="Salma", sanad="Sanad")


def _wrap(s, n):
    out = []
    for para in s.split("\n"):
        out += textwrap.wrap(para, n) or [""]
    return out


def _grad(fig, x, y, w, h, c1, c2, z=0):
    ax = fig.add_axes([x, y, w, h], zorder=z); ax.axis("off")
    cm = LinearSegmentedColormap.from_list("g", [c1, c2])
    ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto",
              cmap=cm, origin="lower", extent=[0, 1, 0, 1])


def _plane(ax, x, y, s, color="white", angle=18):
    pts = np.array([(0, 0), (-0.55, -0.18), (-0.42, -0.05), (-0.95, 0.04),
                    (-0.42, 0.05), (-0.5, 0.30), (-0.32, 0.30), (-0.05, 0.10),
                    (0.55, 0.06), (0.66, 0.0), (0.55, -0.05), (-0.05, -0.06)])
    th = np.radians(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ R.T
    ax.add_patch(mpatches.Polygon(np.c_[pts[:, 0] * s + x, pts[:, 1] * s + y],
                 closed=True, fc=color, ec="none", zorder=20))


def _arc(ax, x0, x1, y, lift, color, lw=1.4):
    t = np.linspace(0, 1, 80)
    ax.plot(x0 + (x1 - x0) * t, y + lift * np.sin(np.pi * t),
            ls=(0, (5, 4)), color=color, lw=lw, zorder=10)
    for xx in (x0, x1):
        ax.add_patch(mpatches.Circle((xx, y), 0.006, fc=color, ec="none", zorder=11))


def _header(fig, kicker, title, speaker):
    _grad(fig, 0, 0.86, 1, 0.14, NAVY, SKY)
    h = fig.add_axes([0, 0.86, 1, 0.14], zorder=2); h.axis("off")
    h.set_xlim(0, 1); h.set_ylim(0, 1)
    _arc(h, 0.60, 0.985, 0.45, 0.36, "#ffffff55", 1.1)
    _plane(h, 0.82, 0.78, 0.05, "#ffffffcc", 20)
    h.text(0.045, 0.78, kicker, color=SAND, fontsize=11.5, fontweight="bold", va="center")
    h.text(0.045, 0.38, title, color="white", fontsize=21, fontweight="bold", va="center")
    h.add_patch(mpatches.FancyBboxPatch((0.80, 0.30), 0.165, 0.42,
                boxstyle="round,pad=0.015,rounding_size=0.06", fc="white", ec="none"))
    h.text(0.8825, 0.51, f"SPEAKER  {speaker}", color=NAVY, fontsize=10.5,
           ha="center", va="center", fontweight="bold")


def _footer(fig, n):
    f = fig.add_axes([0, 0, 1, 0.052], zorder=2); f.axis("off")
    f.set_xlim(0, 1); f.set_ylim(0, 1)
    x0, x1, y = 0.30, 0.74, 0.5
    f.plot([x0, x1], [y, y], color=LINE, lw=1.6, zorder=1)
    for i in range(10):
        xi = x0 + (x1 - x0) * i / 9
        f.add_patch(mpatches.Circle((xi, y), 0.004,
                    fc=(OCEAN if (i + 1) <= n else "#c9d3d8"), ec="none"))
    _plane(f, x0 + (x1 - x0) * (n - 1) / 9, y, 0.018, OCEAN, 0)
    f.text(0.045, 0.5, "IE University · Machine Learning Foundations 2026",
           fontsize=8.5, color=MUTE, va="center")
    f.text(0.955, 0.5, f"BOARDING  {n} / 10", fontsize=9.5, color=OCEAN,
           ha="right", va="center", fontweight="bold")


def _body(fig):
    b = fig.add_axes([0, 0, 1, 1], zorder=1); b.axis("off")
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    return b


def _block(b, x, top, w, tag, col, text, size=10):
    chars = int(w / 0.0072)
    lines = _wrap(text, chars)
    b.add_patch(mpatches.FancyBboxPatch((x, top - 0.030), 0.0135 * len(tag) + 0.02,
                0.030, boxstyle="round,pad=0.004,rounding_size=0.015",
                fc=col, ec="none"))
    b.text(x + 0.010, top - 0.015, tag, color="white", fontsize=8.5,
           fontweight="bold", va="center")
    ty = top - 0.058
    for ln in lines:
        b.text(x, ty, ln, fontsize=size, color=INK, va="top")
        ty -= 0.0335
    return ty - 0.018


def _struggle(b, x, top, w, struggle, fix):
    iw = int((w - 0.05) / 0.0072)
    sl, fl = _wrap("Struggle: " + struggle, iw), _wrap("Fix: " + fix, iw)
    h = 0.044 + (len(sl) + len(fl)) * 0.031
    b.add_patch(mpatches.FancyBboxPatch((x, top - h), w, h,
                boxstyle="round,pad=0.006,rounding_size=0.015",
                fc="#fbf0ec", ec=ALERT, lw=1.1))
    b.text(x + 0.016, top - 0.026, "STRUGGLE  →  HOW WE HANDLED IT",
           color=ALERT, fontsize=8.5, fontweight="bold")
    ty = top - 0.050
    for ln in sl:
        b.text(x + 0.016, ty, ln, fontsize=9.5, color=INK, va="top"); ty -= 0.031
    for ln in fl:
        b.text(x + 0.016, ty, ln, fontsize=9.5, color=GO, va="top",
               fontweight="bold"); ty -= 0.031
    return top - h - 0.018


def _keep(b, x, top, w, text):
    iw = int((w - 0.05) / 0.0072)
    lines = _wrap(text, iw)
    h = 0.044 + len(lines) * 0.030
    b.add_patch(mpatches.FancyBboxPatch((x, top - h), w, h,
                boxstyle="round,pad=0.006,rounding_size=0.015", fc=NAVY, ec="none"))
    b.text(x + 0.016, top - 0.026, "★  FOR THE PROFESSOR", color=SAND,
           fontsize=8.5, fontweight="bold")
    ty = top - 0.050
    for ln in lines:
        b.text(x + 0.016, ty, ln, color="white", fontsize=9.5, va="top"); ty -= 0.030
    return top - h - 0.018


def _img(ax, name):
    p = FIGS / name
    if p.exists():
        ax.imshow(mpimg.imread(p))
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(LINE); s.set_linewidth(1)
    else:
        ax.text(0.5, 0.5, f"[{name}]", ha="center", va="center", fontsize=9, color=MUTE)
    ax.set_xticks([]); ax.set_yticks([])


def _figbox(fig, x, y, w, h, name, cap):
    _img(fig.add_axes([x, y, w, h], zorder=3), name)
    fig.text(x + w / 2, y - 0.026, cap, ha="center", fontsize=9, color=MUTE,
             style="italic")


def _pipeline(b, y):
    st = ["Raw BTS\n6.97 M", "Seeded\n150 k", "Booking\nfeatures", "Train-fold\nencoders",
          "Column\nTransformer", "Ladder +\nisotonic", "τ* profit\ndecision"]
    cols = [OCEAN, OCEAN, SAND, SAND, SKY, NAVY, ALERT]
    x0, x1, n = 0.045, 0.955, len(st)
    gap = (x1 - x0) / n
    w = gap * 0.80
    for i, (s, c) in enumerate(zip(st, cols)):
        cx = x0 + i * gap
        b.add_patch(mpatches.FancyBboxPatch((cx, y), w, 0.10,
                    boxstyle="round,pad=0.004,rounding_size=0.012", fc=c, ec="none"))
        b.text(cx + w / 2, y + 0.05, s, ha="center", va="center", color="white",
               fontsize=8.5, fontweight="bold")
        if i < n - 1:
            b.annotate("", xy=(cx + gap, y + 0.05), xytext=(cx + w, y + 0.05),
                       arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.5))


def _breakeven(b, x, y, w):
    for label, val, c in [("Required confidence  τ*", 0.63, ALERT),
                          ("Best route×carrier cohort", 0.077, SAND),
                          ("EC261-eligible base rate", 0.012, GO)]:
        b.text(x, y + 0.030, label, fontsize=10, color=INK, va="center")
        b.add_patch(mpatches.Rectangle((x, y - 0.022), w * val / 0.63, 0.040,
                    color=c, lw=0))
        b.text(x + w * val / 0.63 + 0.008, y - 0.002,
               f"{val:.0%}" if val >= 0.05 else f"{val:.1%}",
               fontsize=10, color=INK, va="center", fontweight="bold")
        y -= 0.105
    b.text(x, y + 0.02, "≈ 8x gap: arithmetic, not a model failure",
           fontsize=10.5, color=ALERT, fontweight="bold", style="italic")


def build_slides(path):
    FS = (13.333, 7.5)
    with PdfPages(path) as pdf:
        # ---------- S1 boarding-pass title ----------
        fig = plt.figure(figsize=FS)
        _grad(fig, 0, 0, 1, 1, NAVY, OCEAN)
        b = _body(fig)
        _arc(b, 0.08, 0.92, 0.66, 0.12, "#ffffff44", 1.6)
        _plane(b, 0.70, 0.785, 0.065, "#ffffffdd", 16)
        b.text(0.5, 0.605, "Profitable Flight-Delay Prediction Under EC261",
               color="white", fontsize=32, fontweight="bold", ha="center")
        b.text(0.5, 0.53, "A Diagnosed Negative Result", color=SAND,
               fontsize=20, fontweight="bold", ha="center")
        b.text(0.5, 0.45, "Can ML find cheap flights whose EC261 payout beats the "
               "ticket?  We test it as a", color=MIST, fontsize=12.5, ha="center")
        b.text(0.5, 0.42, "machine-learning engineering problem - and prove, from "
               "two data sources, why it cannot.", color=MIST, fontsize=12.5,
               ha="center")
        b.add_patch(mpatches.FancyBboxPatch((0.17, 0.15), 0.66, 0.17,
                    boxstyle="round,pad=0.01,rounding_size=0.03", fc="white", ec="none"))
        b.text(0.205, 0.262, "BOARDING PASS", color=OCEAN, fontsize=10.5,
               fontweight="bold")
        b.text(0.205, 0.205, "Habib · Issam · Adam · Lama · Salma · Sanad",
               color=INK, fontsize=12, fontweight="bold")
        for lx, lab, val in ((0.57, "FLIGHT", "EC261"), (0.665, "GATE", "10"),
                             (0.745, "DURATION", "20 MIN")):
            b.text(lx, 0.262, lab, color=MUTE, fontsize=8, fontweight="bold")
            b.text(lx, 0.205, val, color=NAVY, fontsize=12, fontweight="bold")
        b.text(0.5, 0.075, "IE University · Machine Learning Foundations "
               "(BCSAI2025CSAI.2.M.A C2) · Spring 2026", color="#ffffffaa",
               fontsize=10.5, ha="center")
        pdf.savefig(fig); plt.close(fig)

        def slide(kicker, title, speaker, n, fn):
            fig = plt.figure(figsize=FS)
            fig.add_axes([0, 0, 1, 1], zorder=0).axis("off")
            bd = _body(fig)
            _header(fig, kicker, title, speaker)
            _footer(fig, n)
            fn(fig, bd)
            pdf.savefig(fig); plt.close(fig)

        LX, LW, RX, RW, TOP = 0.045, 0.455, 0.535, 0.42, 0.80

        # ---------- S2 ----------
        def s2(fig, b):
            y = _block(b, LX, TOP, LW, "THE REGULATION", OCEAN,
                "EC261 pays EUR 250-600 when a flight arrives 3h+ late, but only "
                "if the cause is carrier-attributable. Weather, ATC and strikes "
                "are exempt (Article 5(3)).")
            y = _block(b, LX, y, LW, "THE REFRAME - KEY DECISION", SAND,
                "We do not predict 'is it delayed?'. We decide 'is buying this "
                "ticket positive expected value?': buy iff p.aC > T + claim + "
                "travel. A ranking/decision problem, not accuracy.")
            _block(b, LX, y, LW, "WHY ML IS THE RIGHT TOOL", SKY,
                "Worthwhile only if a learnable signal separates payable from "
                "non-payable cheap flights. Testing that honestly - including "
                "that none may exist - is the project.")
            for i, (lx, t, c) in enumerate([
                    (0.55, "Cheap\nticket", OCEAN), (0.70, "Carrier\ndelay 3h+?", SAND),
                    (0.85, "EUR 250-\n600", ALERT)]):
                b.add_patch(mpatches.FancyBboxPatch((lx, 0.66), 0.115, 0.12,
                            boxstyle="round,pad=0.008,rounding_size=0.02", fc=c, ec="none"))
                b.text(lx + 0.0575, 0.72, t, ha="center", va="center",
                       color="white", fontsize=10.5, fontweight="bold")
                if i < 2:
                    b.annotate("", xy=(lx + 0.155, 0.72), xytext=(lx + 0.115, 0.72),
                               arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=2))
            _keep(b, RX, 0.55, RW,
                  "We are graded on rigour and reasoning, not profit. A "
                  "diagnosed negative result scores like a high-F1 model "
                  "(brief sections 1-2).")
            b.text(RX, 0.30, "Hypothesis under test:", fontsize=10.5,
                   color=MUTE, style="italic", fontweight="bold")
            b.text(RX, 0.265, "a cheap ticket on a likely carrier-attributable",
                   fontsize=10.5, color=MUTE, style="italic")
            b.text(RX, 0.235, "delay is a positive-EV bet.", fontsize=10.5,
                   color=MUTE, style="italic")
        slide("WHY THIS PROBLEM", "Problem framing & why it is an ML problem",
              SPK["issam"], 2, s2)

        # ---------- S3 ----------
        def s3(fig, b):
            y = _block(b, LX, TOP, LW, "DATA SOURCES (REAL)", OCEAN,
                "US BTS On-Time + Cause-of-Delay 2024: 6,965,247 real flights "
                "(primary). EUROCONTROL R&D Archive: 3.89 M EU flights, held out "
                "for transfer validation.")
            y = _block(b, LX, y, LW, "KEY LABELLING DECISION", SAND,
                "y = 1 iff arrival-delay >= 180 min AND the dominant cause is "
                "carrier or late-aircraft. Dropping weather/ATC moves the base "
                "rate 1.43% -> 1.18%; raw delay would over-count payouts.")
            _struggle(b, LX, y, LW,
                "6.97 M rows make tuning seven models intractable on a laptop.",
                "a seeded, month x label-stratified 150 k sample (scripted; base "
                "rate preserved at 1.175%).")
            _figbox(fig, RX, 0.20, RW, 0.55, "01_hour_pattern.png",
                    "EC261-eligible rate by hour - real operational structure")
        slide("STEP 1 · DATA ENGINEERING", "Data sourcing & the label decision",
              SPK["issam"], 3, s3)

        # ---------- S4 ----------
        def s4(fig, b):
            y = _block(b, LX, TOP, LW, "CLASS IMBALANCE -> METRICS", OCEAN,
                "~1% positive. Accuracy is meaningless (a constant 'skip' scores "
                "98.8%). Decision: reject accuracy; lead with PR-AUC, ROC-AUC, "
                "Brier, ECE and a money-denominated ROI.")
            y = _block(b, LX, y, LW, "MISSINGNESS & OUTLIERS", SAND,
                "Aircraft age recovered via the FAA join (6.6% null). Distance "
                "retained, not winsorised: EC261 payout is a step function of "
                "distance, so clipping would corrupt the label.")
            _struggle(b, LX, y, LW,
                "Weather features were silently 100% null.",
                "root-caused an un-wired loader; weather now joins for 88.7% of "
                "rows and is used.")
            _figbox(fig, RX, 0.20, RW, 0.55, "01_correlations.png",
                    "Weak feature-target correlation foreshadows the result")
        slide("STEP 2 · EDA & PREPROCESSING",
              "Exploratory analysis & preprocessing decisions", SPK["lama"], 4, s4)

        # ---------- S5 ----------
        def s5(fig, b):
            _block(b, LX, TOP, 0.91, "LEAKAGE CONTROL - DESIGN DECISION", SAND,
                "14-day booking-horizon whitelist · forbidden columns dropped at "
                "step 1 · train-fold-only rolling encoders · "
                "ExpandingTimeSeriesSplit (no future->past leak) · "
                "ColumnTransformer remainder='drop' as defence in depth. The "
                "whole flow is one serializable sklearn Pipeline.")
            _pipeline(b, 0.40)
            b.text(0.5, 0.36, "Preprocessing, feature engineering and the model "
                   "are inseparable and reproducible.", ha="center",
                   fontsize=10, color=MUTE, style="italic")
            _struggle(b, LX, 0.30, LW,
                "Three notebooks had never been run end-to-end.",
                "enforced top-to-bottom execution; found and fixed three "
                "zero-grade bugs before they cost the grade.")
            _keep(b, RX, 0.30, RW, "Everything runs top-to-bottom and is "
                  "reproducible from a seeded script - please run `make` to "
                  "verify.")
        slide("STEP 3 · PIPELINE",
              "Pipeline architecture & leakage control", SPK["lama"], 5, s5)

        # ---------- S6 ----------
        def s6(fig, b):
            y = _block(b, LX, TOP, LW, "MODEL LADDER (JUSTIFIED)", OCEAN,
                "Trivial (Dummy, route-rate) -> Logistic Regression -> Decision "
                "Tree -> Random Forest -> XGBoost -> MLP. Boosted trees dominate "
                "tabular data; the MLP is the required neural baseline.")
            y = _block(b, LX, y, LW, "CALIBRATION + MONEY OBJECTIVE", SAND,
                "Every model is isotonic-calibrated - expected-value maths needs "
                "honest probabilities. We derived the per-flight threshold "
                "tau*(T,d) in closed form and tuned a custom profit scorer.")
            _block(b, LX, y, LW, "TUNING (BUDGET JUSTIFIED)", SKY,
                "Grid (LogReg), Randomized (RF) and Bayesian (XGB) search, each "
                "matched to the model's dimensionality.")
            _figbox(fig, RX, 0.20, RW, 0.55, "04_tau_heatmap.png",
                    "tau* is the break-even probability - already 0.6-1.0+")
        slide("STEP 4 · MODELLING",
              "Model ladder, calibration & the money objective", SPK["adam"], 6, s6)

        # ---------- S7 ----------
        def s7(fig, b):
            y = _block(b, LX, TOP, LW, "METRICS MATCHED TO THE TASK", OCEAN,
                "Imbalanced decision task -> PR-AUC, ROC-AUC, Brier, ECE and a "
                "money ROI. Accuracy explicitly rejected and reported as such.")
            y = _block(b, LX, y, LW, "VALIDATION STRATEGY", SAND,
                "Comparison on BOTH validation and test sets. Expanding "
                "time-series CV; reference RF+iso PR-AUC = 0.031 +/- 0.014 "
                "(95% CI 0.017-0.045). ROC, learning and confusion curves "
                "all reported.")
            _struggle(b, LX, y, LW,
                "Logistic Regression overflowed (matmul) in the learning curve.",
                "root-caused and swapped to a tree-based estimator - fixed the "
                "cause, not just the warning.")
            _figbox(fig, RX, 0.20, RW, 0.55, "05_calibration_before_after.png",
                    "Calibration is decisive - it is what makes models abstain")
        slide("STEP 5 · EVALUATION",
              "Evaluation methodology & rigour", SPK["habib"], 7, s7)

        # ---------- S8 ----------
        def s8(fig, b):
            y = _block(b, LX, TOP, LW, "THE RESULT", OCEAN,
                "Every calibrated model converges on the same decision: abstain. "
                "Test ROI 0%, zero buys, -EUR 21,774. Uncalibrated models bet "
                "and lose EUR 0.3-1.0 M (ROI down to -191.7%).")
            y = _struggle(b, LX, y, LW,
                "ROI was 0 everywhere - it looked like a bug.",
                "derived the closed-form break-even and PROVED it is structural "
                "EC261 arithmetic, not a code fault.")
            _keep(b, LX, y, LW, "The negative result IS the finding - "
                  "diagnosed, not a failure to optimise.")
            _breakeven(b, RX + 0.02, 0.66, 0.36)
            b.text(RX + 0.02, 0.215, "Required confidence (~63%) is ~8x the best",
                   fontsize=10, color=INK)
            b.text(RX + 0.02, 0.185, "cohort rate (7.7%) and ~53x the base rate.",
                   fontsize=10, color=INK)
        slide("THE FINDING", "Result & the structural diagnosis",
              SPK["adam"], 8, s8)

        # ---------- S9 ----------
        def s9(fig, b):
            _block(b, LX, TOP, 0.91, "INDEPENDENT CONFIRMATION - 3.89 M EU FLIGHTS",
                OCEAN,
                "Scoring the BTS-trained model on real EUROCONTROL data: the "
                "ranking is perfectly inverted (Spearman rho = -1.00, top-k lift "
                "<= 1.03). Real EU signal exists (15.6% -> 39.2% by haul) but "
                "the model cannot exploit it - the censored filed-plan label and "
                "domain shift dominate.")
            _figbox(fig, 0.05, 0.135, 0.42, 0.45, "06_decile_monotonicity.png",
                    "EU decile rate falls as model confidence rises (rho = -1.00)")
            _figbox(fig, 0.53, 0.135, 0.42, 0.45, "05_kernel_shap.png",
                    "Permutation + kernel SHAP; cause-cols 0.000 (leakage ✓)")
        slide("STEP 6 · TRANSFER & INTERPRETATION",
              "Transfer validation & model interpretation",
              f'{SPK["sanad"]}+{SPK["salma"]}', 9, s9)

        # ---------- S10 ----------
        def s10(fig, b):
            b.text(LX, 0.815, "\"Negative results are findings - as much insight "
                   "as a high F1 score.\"  (brief section 1)", fontsize=11.5,
                   color=OCEAN, style="italic")
            y = _block(b, LX, 0.78, LW, "LIMITATIONS (HONEST)", OCEAN,
                "Synthetic fares (real fares paywalled) are the largest "
                "external-validity threat - bounded by price-robust break-even "
                "arithmetic. Single-year 2024 split; EUROCONTROL filed-plan "
                "censoring; cancellations out of scope.")
            _block(b, LX, y, LW, "ETHICS & BIAS", SAND,
                "Speculating on others' disruption is legal but ethically "
                "marginal; cheap-fare selection bias; a US-trained model is not "
                "EU-representative (rho = -1.00 shows it).")
            y = _block(b, RX, 0.78, RW, "WHAT WE WOULD CHANGE", SKY,
                "Acquire real fare data and build a two-stage label (delay, then "
                "carrier-attributability). Both flagged as future work.")
            _keep(b, RX, y, RW, "Judged on rigour; code runs top-to-bottom; "
                  "result diagnosed from two independent sources.")
            b.add_patch(mpatches.Rectangle((LX, 0.075), 0.91, 0.085, color="#efe7d6"))
            b.text(LX + 0.015, 0.13, "Conclusion: ML correctly shows the EC261 "
                   "ticket-buying strategy is structurally unprofitable - proven "
                   "from BTS economics AND the EU transfer.", fontsize=11,
                   color=INK, fontweight="bold", va="center")
            b.text(LX + 0.015, 0.098, "Team: Rahal · Arida · Khoury · Moucattash "
                   "· Nsour · ALbilleh   -   all six members present & speaking",
                   fontsize=9.5, color=MUTE, va="center")
        slide("REFLECTION", "Reflection, ethics & for the professor",
              SPK["sanad"], 10, s10)

    print(f"[deliverables] wrote {path}")


def build_poster(path):
    W, H = 841 / 25.4, 594 / 25.4
    f = plt.figure(figsize=(W, H)); f.patch.set_facecolor(PAPER)
    bg = f.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.add_patch(mpatches.Rectangle((0, 0.9), 1, 0.1, color=NAVY))
    bg.text(0.5, 0.957, "Profitable Flight-Delay Prediction Under EC261 - "
            "A Diagnosed Negative Result", color="white", fontsize=30,
            fontweight="bold", ha="center", va="center")
    bg.text(0.5, 0.918, "Rahal · Arida · Khoury · Moucattash · Nsour · ALbilleh "
            "   •   IE University   •   BCSAI2025CSAI.2.M.A C2",
            color=MIST, fontsize=15, ha="center", va="center")
    bg.text(0.5, 0.882, "No positive-EV ticket exists: every calibrated model "
            "abstains (ROI 0%). Required confidence ~63% is ~8x any achievable "
            "delay rate; a 3.89M-flight EUROCONTROL test confirms it (rho = -1.00).",
            color=INK, fontsize=14, ha="center", va="center", style="italic")

    def panel(x, y, w, h, head, lines, color=OCEAN):
        a = f.add_axes([x, y, w, h]); a.axis("off")
        a.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=False, ec=LINE, lw=1.5))
        a.add_patch(mpatches.Rectangle((0, 0.86), 1, 0.14, color=color))
        a.text(0.04, 0.93, head, color="white", fontsize=15, fontweight="bold", va="center")
        a.text(0.04, 0.80, "\n".join(lines), fontsize=12.5, va="top", color=INK)

    def figpanel(x, y, w, h, name, cap):
        a = f.add_axes([x, y, w, h]); _img(a, name)
        a.set_title(cap, fontsize=11, color=MUTE)

    panel(0.03, 0.60, 0.28, 0.27, "MOTIVATION",
          ["EC261 pays EUR 250-600", "for carrier 3h+ delays.", "Cheap ticket + likely",
           "delay = positive-EV bet?", "", "ML as a decision,", "not accuracy."])
    figpanel(0.335, 0.585, 0.32, 0.30, "04_tau_heatmap.png",
             "HERO: required tau*(T,d) ~ 0.63")
    panel(0.69, 0.60, 0.28, 0.27, "DATA",
          ["Real US BTS 2024:", "6.97 M -> seeded 150 k.", "EC261 base rate 1.18%.",
           "EUROCONTROL 3.89 M", "real flights (transfer)."])
    panel(0.03, 0.31, 0.28, 0.26, "METHOD",
          ["Leakage-audited pipeline.", "6-model ladder, isotonic", "calibration.",
           "Grid/Random/Bayes on", "a custom profit scorer."])
    figpanel(0.335, 0.30, 0.32, 0.27, "04_profit_curve.png",
             "No threshold is profitable - abstain")
    panel(0.69, 0.31, 0.28, 0.26, "EU TRANSFER",
          ["Spearman rho = -1.00", "(ranking inverted).", "Top-k lift <= 1.03.",
           "EU signal 15.6->39.2%", "by haul - unusable."], color=ALERT)
    panel(0.03, 0.03, 0.28, 0.25, "RESULTS (test n=29,604)",
          ["All calibrated models:", "ROI 0%, 0 buys.", "Best RF+iso ROC 0.65.",
           "Uncal XGB -191.7%", "-> calibration decisive."])
    figpanel(0.335, 0.025, 0.32, 0.25, "05_calibration_before_after.png",
             "Calibration makes the model abstain")
    panel(0.69, 0.03, 0.28, 0.25, "WHY / REFS",
          ["EC261 cap x a0.65 <", "ticket + EUR 65 for a", "~1% event. Structural.",
           "Negative result =", "finding (brief 1/2)."], color=MUTE)
    f.savefig(path, dpi=200); plt.close(f)
    print(f"[deliverables] wrote {path}")


if __name__ == "__main__":
    build_slides(ROOT / "reports" / "slides.pdf")
    build_poster(ROOT / "reports" / "poster.pdf")
