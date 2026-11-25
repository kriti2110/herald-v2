"""Shared style + data loading for all paper figures/tables. Every number comes from results/*.json|parquet."""
import json, os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

R = os.environ.get("HERALD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = f"{R}/paper/figures"
TAB = f"{R}/paper/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

C = dict(indigo="#4F46E5", teal="#0D9488", coral="#F97316", rose="#E11D48", amber="#F59E0B", slate="#64748B",
         violet="#7C3AED", sky="#0EA5E9", emerald="#10B981", ink="#0F172A", mist="#E2E8F0", pink="#EC4899", lime="#84CC16")
TASK_COL = {"forecast": C["indigo"], "detect": C["teal"], "stay": C["coral"]}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#94A3B8", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#E2E8F0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.2,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "text.color": C["ink"],
    "axes.labelcolor": C["ink"], "xtick.color": "#334155", "ytick.color": "#334155"})

W1, W2 = 3.45, 7.1   # IEEE single / double column width (in)


def panel(ax, s, dx=-0.13, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=C["ink"], va="bottom")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", facecolor="white"); plt.close(fig); print("saved", name)


def J(name):
    return json.load(open(f"{R}/results/{name}"))


TS = J("final_twostage_metrics.json")          # two-stage final model (cv/transfer x forecast/detect)
GB = J("final_metrics.json")                   # GBM / TCN / ensemble + ablations
HZ = J("horizons_12_24.json")
SL = J("stay_level.json")
SD = J("stay_level_diagnostics.json")
R3 = J("round3_val_only.json")
F12 = J("features_v1_vs_v2_val_only.json")
TOP = J("topsis_same_patients.json")
TOPC = pd.read_csv(f"{R}/results/topsis_same_patients_criteria.csv", index_col=0)
AUD2 = J("prefix_audit_v2.json"); AUD1 = J("prefix_audit.json"); AUD3 = J("prefix_audit_v3.json")
COLS = json.load(open(f"{R}/cache/columns2.json"))
FOLDS = pd.read_csv(f"{R}/results/patient_folds.csv")
