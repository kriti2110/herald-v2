"""Rigor scorecard + TOPSIS used by BOTH the figures and the tables (single source of truth).

Rigor criteria follow TRIPOD+AI-style reporting items (Collins et al., BMJ 2024). Each paper was scored by the authors from the
text we could read (0 = not reported, 0.5 = partial / unstated, 1 = reported). The scoring is a judgement and is disclosed as such."""
import numpy as np
import pandas as pd
from figstyle import TS, SL, SD

F2 = lambda k: TS[k]["FINAL_stage2"]

CRIT = ["Patient-level\nsplit stated", "Test at natural\nprevalence", "Uncertainty\nreported", "Leakage / prefix\naudit", "Cross-site or\nexternal validation", "AUROC and AUPRC\nboth reported"]
CRIT_FLAT = [c.replace("\n", " ") for c in CRIT]
PAPERS = ["HERALD (hourly)", "HERALD (stay-level)", "SepsisAI [5]", "LSTM-GAT [6]", "Transformer-eICU [7]"]
SCORES = pd.DataFrame([[1, 1, 1, 1, 0.5, 1],
                       [1, 1, 1, 1, 0.5, 1],
                       [1, 0.5, 0, 0, 0, 1],
                       [0, 0.5, 0.5, 0, 1, 0.5],
                       [0, 0, 0.5, 0, 1, 0]], index=PAPERS, columns=CRIT_FLAT, dtype=float)
SCORES["Rigor score"] = SCORES.mean(1)
# headline AUROC each paper reports under ITS OWN protocol (not comparable; see text)
AUROC = pd.Series({"HERALD (hourly)": F2("cv/detect")["pooled"]["AUROC"], "HERALD (stay-level)": SL["primary_no_length"]["cv_pooled"]["AUROC"],
                   "SepsisAI [5]": 0.95, "LSTM-GAT [6]": 0.853, "Transformer-eICU [7]": 0.99})


def topsis(M, w, cost=None):
    M = np.asarray(M, float); cost = np.zeros(M.shape[1], bool) if cost is None else np.asarray(cost)
    R = M / np.sqrt((M ** 2).sum(0)); V = R * np.asarray(w)
    best = np.where(cost, V.min(0), V.max(0)); worst = np.where(cost, V.max(0), V.min(0))
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1)); return dm / (dp + dm)


LIT_M = np.c_[AUROC.to_numpy(), SCORES["Rigor score"].to_numpy()]
LIT_EQUAL = pd.Series(topsis(LIT_M, [0.5, 0.5]), index=PAPERS)
SWEEP_W = np.linspace(0, 1, 101)
LIT_SWEEP = pd.DataFrame({p: [topsis(LIT_M, [1 - w, w])[i] for w in SWEEP_W] for i, p in enumerate(PAPERS)}, index=SWEEP_W)
RIGOR_ONLY = pd.Series(topsis(SCORES[CRIT_FLAT].to_numpy(), [1 / 6] * 6), index=PAPERS)

if __name__ == "__main__":
    print(SCORES.round(3).to_string()); print("\nAUROC", AUROC.round(3).to_dict()); print("\nTOPSIS equal (AUROC, rigor):\n", LIT_EQUAL.round(3).to_string())
    print("\nrigor-criteria-only TOPSIS:\n", RIGOR_ONLY.round(3).to_string())
    for w in (0.0, 0.25, 0.5, 0.75, 1.0): print(w, LIT_SWEEP.loc[w].round(3).to_dict())
    cross = [(w, LIT_SWEEP.loc[w].idxmax()) for w in SWEEP_W][::10]; print(cross)


# ======================================================================= TOPSIS on PUBLISHED SCORES
import json as _json
_OP = _json.load(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "results", "stay_level_operating_point.json")))
_sl = SL["primary_no_length"]["cv_pooled"]
# (AUROC, AUPRC) as reported by each study under ITS OWN protocol; dagger = quoted second-hand from Mansoor et al. [8]
PERF = pd.DataFrame({
    "SepsisAI [5]": (0.95, 0.96), "MB-TCN [8]†": (0.892, 0.527), "PoEMS [8]†": (0.911, 0.675), "SupMix [8]†": (0.883, 0.667),
    "HERALD hourly": (F2("cv/detect")["pooled"]["AUROC"], F2("cv/detect")["pooled"]["AUPRC"]),
    "HERALD stay (natural)": (_sl["AUROC"], _sl["AUPRC"]),
    "HERALD stay (balanced)": (SD["balanced_432_432_view"]["AUROC"], SD["balanced_432_432_view"]["AUPRC"])}, index=["AUROC", "AUPRC"]).T
PERF["C*"] = topsis(PERF[["AUROC", "AUPRC"]].to_numpy(), [0.5, 0.5])
# (AUROC, sensitivity/recall, F1) -- only studies that report all three
OPS = pd.DataFrame({
    "SepsisAI [5]": (0.95, 0.8819, 0.9214), "LSTM-GAT [6]": (0.853, 0.706, 0.627), "Transformer-eICU [7]": (0.99, 0.967, 0.959),
    "HERALD stay (balanced)": (SD["balanced_432_432_view"]["AUROC"], _OP["sens"], _OP["f1_bal"])}, index=["AUROC", "Sensitivity", "F1"]).T
OPS["C*"] = topsis(OPS[["AUROC", "Sensitivity", "F1"]].to_numpy(), [1 / 3] * 3)

if __name__ == "__main__":
    print("\nPERF (AUROC, AUPRC):\n", PERF.round(3).sort_values("C*", ascending=False).to_string())
    print("\nOPS (AUROC, sens, F1):\n", OPS.round(3).sort_values("C*", ascending=False).to_string())
