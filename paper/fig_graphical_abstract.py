"""Graphical abstract (Elsevier: landscape, >=1328x531 px). All numbers read from results files / numbers.tex values."""
import re, json
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
num = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", open(f"{R}/paper/numbers.tex").read(), re.M))
fig, ax = plt.subplots(figsize=(13.3, 5.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off"); ax.grid(False)
def box(x, y, w, h, title, lines, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=col, lw=2))
    ax.text(x + w/2, y + h - 2.2, title, ha="center", va="top", fontsize=13, fontweight="bold", color=col)
    for i, l in enumerate(lines): ax.text(x + w/2, y + h - 6.6 - i*3.1, l, ha="center", va="top", fontsize=10.5, color=C["ink"])
box(1, 9, 22, 22, "1  Causal features", ["370 backward-looking", "features per ICU hour", "forward-fill, trailing", "windows, no look-ahead"], C["indigo"])
box(26, 9, 22, 22, "2  Prefix audit", [f"{num['NCmp'].replace('{,}',',')} comparisons:", "no violations", f"leaky control flagged", f"in {num['NCtrl'].replace('{,}',',')} cuts"], C["teal"])
box(51, 9, 22, 22, "3  Two-stage model", ["early prediction (6 h)", "hourly detection", "patient-level 5-fold CV", "natural prevalence"], C["coral"])
box(76, 9, 23, 22, "4  Honest evaluation", [f"AUROC {num['AurocFc']} / {num['AurocDt']}", f"other hospital: {num['AurocFcT']} / {num['AurocDtT']}", "subgroups, decision curves,", "alert burden, calibration"], C["rose"])
for x in (23.6, 48.6, 73.6): ax.add_patch(FancyArrowPatch((x, 20), (x + 2.2, 20), arrowstyle="-|>", mutation_scale=18, color=C["slate"], lw=2))
ax.text(50, 38.6, "HERALD: prefix-valid, audited sepsis prediction on PhysioNet/CinC 2019", ha="center", va="center", fontsize=16, fontweight="bold")
ax.text(50, 2.2, "A reproducible, leakage-audited evaluation protocol with honest uncertainty, not a claim of state-of-the-art accuracy", ha="center", va="center", fontsize=11, style="italic", color=C["slate"])
fig.savefig(f"{R}/paper/graphical_abstract.png", dpi=200, bbox_inches=None, pad_inches=0); print("ok")
