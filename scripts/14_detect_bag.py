"""ONE pre-specified attempt: detection, bag of 3 boosted configs (B,C,D) on the audited v2 features, logit-averaged.
Same patient folds / fit-val split as 03_gbm.py; early stop on val logloss; test touched once; report whatever it gives."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exec(open(f"{R}/scripts/06_report.py").read().split("final = {}")[0])      # metrics(), fast_boot(), balanced_patient()
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); X = np.load(f"{R}/cache/X2.npy"); T = meta.det_target.to_numpy()
pc, upid = pd.factorize(meta[PID]); f = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]; fold_of, ever = f.fold.to_numpy(), f.ever_septic.to_numpy()
CFGS = {"B": dict(num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
        "C": dict(num_leaves=15, min_child_samples=1000, feature_fraction=.3, lambda_l2=10),
        "D": dict(num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)}
base = dict(objective="binary", metric=["binary_logloss", "auc"], learning_rate=.05, bagging_fraction=.8, bagging_freq=1, max_bin=127, verbose=-1, num_threads=10)
preds = []
for k in range(5):
    tp = np.flatnonzero(fold_of != k); te = np.flatnonzero(fold_of == k)
    a, v = train_test_split(tp, test_size=0.25, stratify=ever[tp], random_state=0); v = train_test_split(v, test_size=0.5, stratify=ever[v], random_state=0)[0]
    Ia, Iv, It = [np.flatnonzero(np.isin(pc, p)) for p in (a, v, te)]
    d = lgb.Dataset(X[Ia], T[Ia]); dv = lgb.Dataset(X[Iv], T[Iv], reference=d); zs = []
    for nm, c in CFGS.items():
        b = lgb.train({**base, **c, "seed": 0}, d, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        zs.append(b.predict(X[It], num_iteration=b.best_iteration, raw_score=True)); print(k, nm, b.best_iteration, round(AUC(T[It], zs[-1]), 4), flush=True)
    z = np.mean(zs, 0); preds.append(pd.DataFrame({PID: upid[pc[It]], "ICULOS": meta.ICULOS.to_numpy()[It], "y": T[It], "z": z, "fold": k})); print("fold", k, "bag", round(AUC(T[It], z), 4), round(AP(T[It], z), 4), flush=True)
d = pd.concat(preds); m = metrics(d, "z", with_ci=True); pf = [AUC(g.y, g.z) for _, g in d.groupby("fold")]
m.update({"fold_AUROC_mean": float(np.mean(pf)), "fold_AUROC_sd": float(np.std(pf, ddof=1)), "balanced_patient_view": balanced_patient(d, "z")})
d.to_parquet(f"{R}/results/detect_bag_preds.parquet"); json.dump(m, open(f"{R}/results/detect_bag.json", "w"), indent=2); print("POOLED", json.dumps({k: m[k] for k in ["AUROC", "AUROC_CI95", "AUPRC", "AUPRC_CI95"]}), "DONE", flush=True)
