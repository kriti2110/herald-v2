import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from analysis import PERF, OPS


def fig20():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(wspace=0.42, width_ratios=[1.3, 1]))
    ax = axs[0]; P = PERF.sort_values("C*"); isH = ["HERALD" in n for n in P.index]; second = ["\u2020" in n for n in P.index]
    cols = [C["violet"] if "balanced" in n else C["indigo"] if h else "#CBD5E1" for n, h in zip(P.index, isH)]
    bars = ax.barh(range(len(P)), P["C*"], color=cols, height=0.66)
    for b, s2 in zip(bars, second):
        if s2: b.set_hatch("////"); b.set_edgecolor("#94A3B8"); b.set_linewidth(0.0)
    for yi, (n, r) in enumerate(P.iterrows()):
        ax.text(r["C*"] + 0.02, yi, f"{r['C*']:.3f}", va="center", fontsize=7.3, fontweight="bold", color=C["indigo"] if "HERALD" in n else C["slate"])
        ax.text(1.80, yi, f"{r['AUROC']:.3f} / {r['AUPRC']:.3f}", va="center", ha="right", fontsize=6.8, color=C["ink"])
    ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index, fontsize=6.9); ax.set_xlim(0, 1.80); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis="y", visible=False)
    ax.text(1.80, len(P) - 0.35, "AUROC / AUPRC", ha="right", fontsize=6.4, fontweight="bold", color=C["slate"]); ax.set_xlabel("TOPSIS closeness  C*  (AUROC, AUPRC; equal weights)", x=0.3); panel(ax, "(a)", -0.55, 1.04)
    ax = axs[1]; O = OPS.sort_values("C*", ascending=False); x = np.arange(len(O)); w = 0.26
    for k, (m, c_) in enumerate((("AUROC", C["indigo"]), ("Sensitivity", C["teal"]), ("F1", C["coral"]))):
        ax.bar(x + (k - 1) * w, O[m], w * 0.94, color=c_, label=m, alpha=0.95)
        for xi, v in zip(x + (k - 1) * w, O[m]): ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.5, rotation=90, va="bottom")
    for xi, v in zip(x, O["C*"]): ax.text(xi, 1.2, f"C* {v:.2f}", ha="center", fontsize=7.2, fontweight="bold", color=C["ink"])
    ax.set_xticks(x); ax.set_xticklabels([n.replace("HERALD stay (balanced)", "HERALD\nstay (bal.)").replace("Transformer-eICU [7]", "Transformer\neICU [7]").replace("SepsisAI [5]", "SepsisAI\n[5]").replace("LSTM-GAT [6]", "LSTM-GAT\n[6]") for n in O.index], fontsize=6.3)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.legend(loc="upper center", ncol=3, fontsize=6.4, bbox_to_anchor=(0.5, -0.2), handlelength=1.0, columnspacing=1.0); ax.grid(axis="x", visible=False); panel(ax, "(b)", -0.16, 1.04)
    save(fig, "fig20_lit_scores_topsis")


if __name__ == "__main__":
    fig20()
