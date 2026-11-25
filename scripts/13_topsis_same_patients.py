"""TOPSIS of HERALD-v2 vs the earlier HERALD models on the IDENTICAL test patients and rows (6-h forecast task).
Criteria (all recomputed here with one code path): AUROC, AUPRC, patient AUROC, Sens@90%-specificity (benefit), alerts-per-TP (cost).
Equal weights (the original paper's weights were not archived); weight sensitivity shown, not used to choose."""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
H = os.environ.get("HERALD_V1_ROOT", "../herald-sepsis"); V = os.environ.get("HERALD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
test = set(pd.read_csv(f"{H}/paper_extras/split_patient_ids.csv").query("split=='test'").patient_id)
mine = pd.read_parquet(f"{V}/results/final_cv_forecast_preds.parquet"); mine = mine[mine.patient_id.isin(test)].reset_index(drop=True)
models = {"LightGBM": "lgbm_extra", "LightGBM-reg": "lgbm_extra_reg", "LightGBM no-meas.": "lgbm_extra_nomeas", "Causal enc.+aux": "causal_aux_s1", "Causal encoder": "causal_s1"}
def crit(y, p, pid):
    pat = pd.DataFrame({"pid": pid, "y": y, "p": p}).groupby("pid").agg(y=("y", "max"), p=("p", "max"))
    thr = np.quantile(p[y == 0], 0.90); tp = int(((p > thr) & (y == 1)).sum()); fp = int(((p > thr) & (y == 0)).sum())
    return {"AUROC": AUC(y, p), "AUPRC": AP(y, p), "PatAUROC": AUC(pat.y, pat.p), "Sens90": tp / y.sum(), "AlertsPerTP": (tp + fp) / max(tp, 1)}
rows = {}
for name, tag in models.items():
    d = pd.read_parquet(f"{H}/outputs/experiments/{tag}/pred_test.parquet")
    m = d.merge(mine[["patient_id", "ICULOS", "y"]], on=["patient_id", "ICULOS"], how="inner")
    assert len(m) == len(d) == len(mine), (name, len(m), len(d), len(mine)); assert (m.target.to_numpy() == m.y.to_numpy()).all()
    rows[name] = crit(d.target.to_numpy(), d.prob.to_numpy(), d.patient_id.to_numpy())
rows["HERALD-v2 (this work)"] = crit(mine.y.to_numpy(), mine.z2.to_numpy(), mine.patient_id.to_numpy())
df = pd.DataFrame(rows).T; print(f"identical rows: {len(mine)}, positives {int(mine.y.sum())}, test patients {mine.patient_id.nunique()}\n"); print(df.round(4).to_string())
def topsis(M, w, cost):
    R = M / np.sqrt((M ** 2).sum(0)); Vm = R * w; best = np.where(cost, Vm.min(0), Vm.max(0)); worst = np.where(cost, Vm.max(0), Vm.min(0))
    dp = np.sqrt(((Vm - best) ** 2).sum(1)); dm = np.sqrt(((Vm - worst) ** 2).sum(1)); return dm / (dp + dm)
M = df.to_numpy(float); cost = np.array([False, False, False, False, True])
W = {"equal": [.2] * 5, "AUROC-heavy": [.4, .15, .15, .15, .15], "AUPRC-heavy": [.15, .4, .15, .15, .15], "operating-point-heavy": [.1, .1, .1, .35, .35]}
out = {}
for k, w in W.items():
    c = pd.Series(topsis(M, np.array(w), cost), index=df.index); out[k] = c.round(4).to_dict()
    print(f"\nTOPSIS C* [{k}]"); print(c.sort_values(ascending=False).round(4).to_string())
df.round(5).to_csv(f"{V}/results/topsis_same_patients_criteria.csv"); json.dump(out, open(f"{V}/results/topsis_same_patients.json", "w"), indent=2)
