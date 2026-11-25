import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve

F2 = lambda k: TS[k]["FINAL_stage2"]


# ---------------------------------------------------------------- Fig 1: pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(-10, 44); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, txt, col, fs=5.5, tc="white", dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.6", fc=("white" if dashed else col), ec=(col if dashed else "none"), lw=1.4, ls=("--" if dashed else "-")))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=(col if dashed else tc), fontweight="bold", linespacing=1.45)

    def arr(x1, y1, x2, y2, col=C["slate"], rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.25, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    W_, H_ = 18.5, 14
    box(0.5, 26, W_, H_, "RAW ICU STREAM\n40,336 stays\n1.55 M hourly rows\nPhysioNet/CinC 2019", C["slate"])
    box(23.5, 26, W_, H_, "CAUSAL FEATURES\nLOCF · trailing windows\nexpanding stats · deltas\n370 features", C["indigo"])
    box(46.5, 26, W_, H_, "STAGE 1 · LightGBM\n4 inner patient folds\nout-of-fold logits", C["teal"])
    box(69.5, 26, 14, H_, "STAGE 2\nscore-history\nstacking", C["violet"])
    box(87, 33, 12.6, 7.4, "EARLY\nPREDICTION 6 h", C["coral"], fs=5.3)
    box(87, 24.6, 12.6, 7.4, "HOURLY\nDETECTION", C["amber"], fs=5.3)
    box(23.5, 3, W_, H_, "STAY AGGREGATOR\nmax · min · mean · last\nno length features", C["emerald"])
    box(46.5, 3, W_, H_, "STAY CLASSIFIER\nLightGBM on 1,208\nstay-level features", C["pink"])
    box(69.5, 3, 14, H_, "RETROSPECTIVE\nSTAY-LEVEL\nDETECTION", C["rose"])
    box(87, 3, 12.6, H_, "CAUSAL TCN\nbenchmarked\n(no gain,\nnot in final)", C["sky"], fs=5.0, dashed=True)
    for a_, b_ in [(19.2, 23.2), (42.2, 46.2), (65.2, 69.2)]:
        arr(a_, 33, b_, 33)
    arr(83.7, 35.5, 86.7, 36.5); arr(83.7, 30.5, 86.7, 28.5)
    arr(10, 25.6, 10, 10, C["emerald"]); arr(10, 10, 23.2, 10, C["emerald"])
    ax.text(11.2, 18, "stay-level path", fontsize=5.4, color=C["emerald"], style="italic")
    arr(42.2, 10, 46.2, 10, C["emerald"]); arr(65.2, 10, 69.2, 10, C["emerald"])
    ax.plot([32.7, 32.7, 93.3], [25.6, 21.0, 21.0], color=C["sky"], lw=1.0, ls=(0, (4, 3)), solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=(93.3, 17.4), xytext=(93.3, 21.0), arrowprops=dict(arrowstyle="-|>", color=C["sky"], lw=1.0, shrinkA=0, shrinkB=0))
    ax.text(63, 22.0, "TCN trained on the same inputs", fontsize=5.2, color=C["sky"], style="italic", ha="center")
    ax.add_patch(FancyBboxPatch((0.5, -9.2), 99, 5.8, boxstyle="round,pad=0.2,rounding_size=1.4", fc="#F1F5F9", ec=C["mist"], lw=1))
    ax.text(50, -6.3, "Protocol: patient-level 5-fold CV  ·  hospital A ↔ B transfer  ·  no test-set resampling  ·  prefix-invariance audit: 370/370 features pass", ha="center", va="center", fontsize=5.9, color=C["ink"])
    save(fig, "fig01_pipeline")


# ---------------------------------------------------------------- Fig 2: cohort
def fig2():
    fig = plt.figure(figsize=(W2, 2.5)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.55)
    inner = gs[0, 0].subgridspec(1, 2, wspace=0.05)
    for i, h in enumerate("AB"):
        ax = fig.add_subplot(inner[0, i]); d = FOLDS[FOLDS.hospital == h]; s_ = int(d.ever_septic.sum()); n = len(d)
        ax.pie([n - s_, s_], colors=[C["mist"], C["coral"]], startangle=90, counterclock=False, wedgeprops=dict(width=0.36, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.1, f"Hospital {h}", ha="center", fontsize=7, fontweight="bold"); ax.text(0, -0.2, f"{n:,} stays", ha="center", fontsize=6.3, color=C["slate"])
        ax.text(0, -1.42, f"{s_:,} septic\n({100 * s_ / n:.1f}%)", ha="center", fontsize=6.8, color=C["coral"], fontweight="bold")
        if i == 0: panel(ax, "(a)", -0.05, 1.02)
    ax = fig.add_subplot(gs[0, 1]); panel(ax, "(b)", -0.30, 1.04)
    items = [("Stay-level (stays)", SL["primary_no_length"]["cv_pooled"]["prevalence"] * 100, C["coral"]),
             ("Hourly detection", F2("cv/detect")["pooled"]["prevalence"] * 100, C["teal"]),
             ("Forecast 24 h", HZ["horizon_24h"]["prevalence"] * 100, C["indigo"]),
             ("Forecast 12 h", HZ["horizon_12h"]["prevalence"] * 100, C["indigo"]),
             ("Forecast 6 h", F2("cv/forecast")["pooled"]["prevalence"] * 100, C["indigo"])]
    y = np.arange(len(items)); ax.barh(y, [v for _, v, _ in items], color=[c for *_, c in items], height=0.62, alpha=0.92)
    for yi, (_, v, _) in zip(y, items): ax.text(v + 0.12, yi, f"{v:.2f}%", va="center", fontsize=7.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([l for l, *_ in items]); ax.set_xlabel("Positive rate of the evaluated units (%)"); ax.set_xlim(0, 8.8); ax.grid(axis="y", visible=False)
    save(fig, "fig02_cohort")


# ---------------------------------------------------------------- Fig 3: audit
def fig3():
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    rows = [("306 features", 0.0, C["teal"]), ("370 features (used)", 0.0, C["teal"]), ("410 features", 0.0, C["teal"]),
            ("Negative control:\ncentered window", 100 * AUD2["negative_control_LEAKY_HR_centered5_violations"] / AUD2["n_prefix_cuts"], C["rose"])]
    y = np.arange(len(rows))[::-1]
    for yi, (l, v, c) in zip(y, rows):
        ax.hlines(yi, 0, v, color=c, lw=3.2, alpha=0.9); ax.scatter(v, yi, s=95, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(v + 4.5, yi, f"{v:.1f}% of cuts" if v > 0 else "0 violations", va="center", fontsize=7.4, fontweight="bold", color=c)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(-3, 135); ax.set_xlabel("Prefix cuts that violate invariance (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{AUD2['n_prefix_cuts']:,} cuts · {AUD2['n_patients']} stays", fontsize=7.2, color=C["slate"], fontweight="normal")
    save(fig, "fig03_audit")


# ---------------------------------------------------------------- Fig 4: main results forest + AUPRC
def main_rows():
    sl, st = SL["primary_no_length"]["cv_pooled"], SL["primary_no_length"]["transfer_pooled"]
    return [("Stay-level · hospital A↔B", st, "stay"), ("Stay-level · 5-fold CV", sl, "stay"),
            ("Detection hourly · hospital A↔B", F2("transfer/detect")["pooled"], "detect"), ("Detection hourly · 5-fold CV", F2("cv/detect")["pooled"], "detect"),
            ("Forecast 6 h · hospital A↔B", F2("transfer/forecast")["pooled"], "forecast"), ("Forecast 6 h · 5-fold CV", F2("cv/forecast")["pooled"], "forecast")]


def fig4():
    rows = main_rows(); fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(wspace=0.08, width_ratios=[1.12, 1])); y = np.arange(len(rows))
    ax = axs[0]; panel(ax, "(a)", -0.52, 1.04)
    for yi, (l, m, t) in zip(y, rows):
        lo, hi = m["AUROC_CI95"]; ax.hlines(yi, lo, hi, color=TASK_COL[t], lw=3.4, alpha=0.45); ax.scatter(m["AUROC"], yi, s=70, color=TASK_COL[t], zorder=3, edgecolor="white", linewidth=1.3)
        ax.text(hi + 0.007, yi, f"{m['AUROC']:.3f}", va="center", fontsize=7.4, fontweight="bold", color=TASK_COL[t])
    ax.axvline(0.5, color=C["slate"], lw=0.8, ls=":"); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.72, 1.0); ax.set_xlabel("AUROC (95% patient-clustered CI)"); ax.grid(axis="y", visible=False)
    ax = axs[1]; panel(ax, "(b)", -0.06, 1.04)
    for yi, (l, m, t) in zip(y, rows):
        lo, hi = m["AUPRC_CI95"]; ax.barh(yi, m["AUPRC"], color=TASK_COL[t], alpha=0.9, height=0.58); ax.errorbar(m["AUPRC"], yi, xerr=[[m["AUPRC"] - lo], [hi - m["AUPRC"]]], color=C["ink"], lw=0.9, capsize=2)
        ax.scatter(m["prevalence"], yi, marker="D", s=26, color="white", edgecolor=C["ink"], zorder=4, linewidth=1)
        ax.text(max(hi, m["AUPRC"]) + 0.012, yi, f"{m['AUPRC']:.3f} (×{m['AUPRC'] / m['prevalence']:.1f})", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlim(0, 1.0); ax.set_xlabel("AUPRC   ◇ = prevalence baseline   (×) = lift"); ax.grid(axis="y", visible=False)
    save(fig, "fig04_main_results")


# ---------------------------------------------------------------- Fig 5: ROC / PR
def fig5():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(wspace=0.22))
    for task, col, lab in (("forecast", C["indigo"], "Early prediction 6 h"), ("detect", C["teal"], "Hourly detection")):
        d = pd.read_parquet(f"{R}/results/final_cv_{task}_preds.parquet", columns=["y", "z2"]); y, s = d.y.to_numpy(), d.z2.to_numpy()
        fpr, tpr, _ = roc_curve(y, s); ix = np.unique(np.r_[np.linspace(0, len(fpr) - 1, 900).astype(int)]); axs[0].plot(fpr[ix], tpr[ix], color=col, lw=2.0, label=f"{lab}  (AUROC {F2('cv/' + task)['pooled']['AUROC']:.3f})")
        axs[0].fill_between(fpr[ix], 0, tpr[ix], color=col, alpha=0.07)
        p, r, _ = precision_recall_curve(y, s); ix = np.linspace(0, len(p) - 1, 1200).astype(int); axs[1].plot(r[ix], p[ix], color=col, lw=2.0, label=f"{lab}  (AUPRC {F2('cv/' + task)['pooled']['AUPRC']:.3f})")
        axs[1].axhline(y.mean(), color=col, lw=0.9, ls="--", alpha=0.7)
    axs[0].plot([0, 1], [0, 1], color=C["slate"], lw=0.9, ls=":"); axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate"); axs[0].legend(loc="lower right"); panel(axs[0], "(a)")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0, 0.45); axs[1].legend(loc="upper right"); panel(axs[1], "(b)")
    axs[1].text(0.98, 0.50, "dashed = prevalence baseline", transform=axs[1].transAxes, ha="right", fontsize=6.5, color=C["slate"])
    save(fig, "fig05_roc_pr")


# ---------------------------------------------------------------- Fig 7: hospital transfer dumbbells
def fig7():
    fig, ax = plt.subplots(figsize=(W2, 2.55)); T = [("Stay-level detection", SL["primary_no_length"]["cv_pooled"]["AUROC"], SL["primary_no_length"]["transfer_A_to_B"]["AUROC"], SL["primary_no_length"]["transfer_B_to_A"]["AUROC"], "stay"),
        ("Hourly detection", F2("cv/detect")["pooled"]["AUROC"], F2("transfer/detect")["per_group"]["A->B"]["AUROC"], F2("transfer/detect")["per_group"]["B->A"]["AUROC"], "detect"),
        ("Early prediction 6 h", F2("cv/forecast")["pooled"]["AUROC"], F2("transfer/forecast")["per_group"]["A->B"]["AUROC"], F2("transfer/forecast")["per_group"]["B->A"]["AUROC"], "forecast")]
    for i, (l, cv, ab, ba, t) in enumerate(T):
        lo, hi = min(ab, ba), cv; ax.hlines(i, lo, hi, color=TASK_COL[t], lw=5, alpha=0.28)
        ax.scatter(cv, i, s=120, color=TASK_COL[t], zorder=3, edgecolor="white", linewidth=1.5, label="Same-hospital CV" if i == 0 else None)
        ax.scatter(ab, i, s=80, marker="^", color=C["ink"], zorder=3, edgecolor="white", linewidth=1.2, label="Train A → test B" if i == 0 else None)
        ax.scatter(ba, i, s=80, marker="s", color=C["slate"], zorder=3, edgecolor="white", linewidth=1.2, label="Train B → test A" if i == 0 else None)
        ax.text(cv, i + 0.27, f"{cv:.3f}", fontsize=7.3, fontweight="bold", color=TASK_COL[t], ha="center"); ax.text(ab, i - 0.30, f"{ab:.3f}", fontsize=6.8, ha="left"); ax.text(ba, i + 0.27, f"{ba:.3f}", fontsize=6.8, ha="center", color=C["slate"])
        ax.text(0.998, i, f"drop ≈ {100 * (cv - np.mean([ab, ba])):.1f} pts", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.4, color=C["rose"], fontweight="bold")
    ax.set_yticks(range(3)); ax.set_yticklabels([t[0] for t in T]); ax.set_ylim(-0.7, 2.6); ax.set_xlim(0.72, 1.08); ax.set_xlabel("AUROC"); ax.grid(axis="y", visible=False); ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.14))
    save(fig, "fig07_transfer")


# ---------------------------------------------------------------- Fig 8: horizons
def fig8():
    H = [(6, TS["cv/forecast"]["stage1"]["pooled"]), (12, HZ["horizon_12h"]), (24, HZ["horizon_24h"])]
    fig, ax = plt.subplots(figsize=(W1, 2.7)); x = np.arange(3); ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color("#94A3B8")
    ax2.bar(x, [m["AUPRC"] for _, m in H], width=0.5, color=C["amber"], alpha=0.55, label="AUPRC (right)")
    for xi, (_, m) in zip(x, H): ax2.text(xi, m["AUPRC"] + 0.006, f"{m['AUPRC']:.3f}", ha="center", fontsize=7.2, fontweight="bold", color="#B45309"); ax2.text(xi, 0.004, f"prev {100 * m['prevalence']:.1f}%", ha="center", fontsize=6.4, color="#7C2D12")
    ax.errorbar(x, [m["AUROC"] for _, m in H], yerr=[[m["AUROC"] - m["AUROC_CI95"][0] for _, m in H], [m["AUROC_CI95"][1] - m["AUROC"] for _, m in H]], color=C["indigo"], lw=2.2, marker="o", ms=7, capsize=3, label="AUROC (left)", zorder=5)
    for xi, (_, m) in zip(x, H): ax.text(xi, m["AUROC"] + 0.012, f"{m['AUROC']:.3f}", ha="center", fontsize=7.2, fontweight="bold", color=C["indigo"])
    ax.set_ylim(0.50, 0.87); ax2.set_ylim(0, 0.32); ax.set_xticks(x); ax.set_xticklabels([f"{h} h" for h, _ in H]); ax.set_xlabel("Prediction horizon"); ax.set_ylabel("AUROC"); ax2.set_ylabel("AUPRC")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.13))
    save(fig, "fig08_horizon")


# ---------------------------------------------------------------- Fig 9: ablations
def fig9():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.65), gridspec_kw=dict(wspace=0.22, width_ratios=[1.25, 1]))
    var = [("full", "All features", C["indigo"]), ("no_care", "− care-pathway (ICULOS, admission time, unit)", C["amber"]), ("physio_only", "− care-pathway − measurement timing", C["rose"])]
    for ai, (mode, ttl, vs) in enumerate((("cv", "(a) Same-hospital 5-fold CV", var), ("transfer", "(b) Cross-hospital transfer", [var[0], var[2]]))):
        ax = axs[ai]; ax.set_title(ttl, fontsize=8.2); w = 0.8 / len(vs)
        for ti, (task, tl) in enumerate((("forecast", "Early prediction 6 h"), ("detect", "Hourly detection"))):
            ab = GB[f"ablation/{mode}/{task}"]
            for vi, (k, l, c) in enumerate(vs):
                v = ab[k]["pooled"]["AUROC"]; xx = ti + (vi - (len(vs) - 1) / 2) * w; ax.bar(xx, v, width=w * 0.92, color=c, alpha=0.93, label=l if ti == 0 else None)
                ax.text(xx, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.6, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Early prediction 6 h", "Hourly detection"]); ax.set_ylim(0.66, 0.90); ax.set_ylabel("AUROC" if ai == 0 else ""); ax.grid(axis="x", visible=False)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=7)
    save(fig, "fig09_ablation")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig7, fig8, fig9):
        f()
