"""Longer prediction horizons (onset within next h hours), same causal v2 features, same patient-level 5-fold CV.
Rows = pre-onset rows (septic) + all rows (non-septic); target = onset within h hours. Config chosen earlier (B), early stop on val logloss."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID, forecast_rows
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(f"{R}/scripts/06_report.py").read().split("final = {}")[0]; exec(src)   # fast_boot, balanced_patient helpers
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); X = np.load(f"{R}/cache/X2.npy")
folds = pd.read_csv(f"{R}/results/patient_folds.csv"); pc, upid = pd.factorize(meta[PID]); f = folds.set_index(PID).loc[upid]
fold_of, ever = f.fold.to_numpy(), f.ever_septic.to_numpy(); keep = meta.fc_keep.to_numpy()
P = dict(objective="binary", metric=["binary_logloss", "auc"], learning_rate=.05, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10,
         bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=10)
out = {}
for h in (12, 24):
    _, T = forecast_rows(meta, h); preds = []
    for k in range(5):
        tp = np.flatnonzero(fold_of != k); te = np.flatnonzero(fold_of == k)
        a, v = train_test_split(tp, test_size=0.25, stratify=ever[tp], random_state=0); v = train_test_split(v, test_size=0.5, stratify=ever[v], random_state=0)[0]
        rows = lambda p: np.flatnonzero(np.isin(pc, p) & keep)
        Ia, Iv, It = rows(a), rows(v), rows(te)
        d = lgb.Dataset(X[Ia], T[Ia]); b = lgb.train(P, d, 3000, valid_sets=[lgb.Dataset(X[Iv], T[Iv], reference=d)], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        z = b.predict(X[It], num_iteration=b.best_iteration, raw_score=True)
        preds.append(pd.DataFrame({PID: upid[pc[It]], "y": T[It], "z": z, "fold": k})); print(h, k, round(AUC(T[It], z), 4), round(AP(T[It], z), 4), flush=True)
    d = pd.concat(preds); m = metrics(d, "z", with_ci=True)
    pf = [AUC(g.y, g.z) for _, g in d.groupby("fold")]; pa = [AP(g.y, g.z) for _, g in d.groupby("fold")]
    m.update({"fold_AUROC_mean": float(np.mean(pf)), "fold_AUROC_sd": float(np.std(pf, ddof=1)), "fold_AUPRC_mean": float(np.mean(pa)), "balanced_patient_view": balanced_patient(d, "z")})
    out[f"horizon_{h}h"] = m; print("POOLED", h, {k: m[k] for k in ["AUROC", "AUROC_CI95", "AUPRC", "AUPRC_CI95", "prevalence", "patient_AUROC"]}, flush=True)
json.dump(out, open(f"{R}/results/horizons_12_24.json", "w"), indent=2); print("DONE")
