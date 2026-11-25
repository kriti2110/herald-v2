"""Hyper-parameter selection on VALIDATION patients only (fold-0 training
portion). Test folds are never read here."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); X = np.load(f"{R}/cache/X.npy")
folds = pd.read_csv(f"{R}/results/patient_folds.csv")
pc, upid = pd.factorize(meta[PID]); fold_of = folds.set_index(PID).loc[upid, "fold"].to_numpy()
ever = folds.set_index(PID).loc[upid, "ever_septic"].to_numpy()
train_pat = np.flatnonzero(fold_of != 0)       # fold 0 = test, never used here
tr, va = train_pat[train_test_split(np.arange(len(train_pat)), test_size=0.25, stratify=ever[train_pat], random_state=0)[0]], None
tr_i, va_i = train_test_split(np.arange(len(train_pat)), test_size=0.25, stratify=ever[train_pat], random_state=0)
tr_pat, va_pat = train_pat[tr_i], train_pat[va_i]
va_pat = train_test_split(va_pat, test_size=0.5, stratify=ever[va_pat], random_state=0)[0]  # same val as 03_gbm.py
CFG = {
 "A_base":   dict(learning_rate=.05, num_leaves=63, min_child_samples=200,  feature_fraction=.5, lambda_l2=1),
 "B":        dict(learning_rate=.03, num_leaves=31, min_child_samples=500,  feature_fraction=.3, lambda_l2=10),
 "C":        dict(learning_rate=.03, num_leaves=15, min_child_samples=1000, feature_fraction=.3, lambda_l2=10),
 "D_extra":  dict(learning_rate=.03, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True),
 "E_small":  dict(learning_rate=.03, num_leaves=7,  min_child_samples=1000, feature_fraction=.3, lambda_l2=10),
}
out = {}
for task in ["forecast", "detect"]:
    T = meta["fc_target"].to_numpy() if task == "forecast" else meta["det_target"].to_numpy()
    def rows(p):
        m = np.zeros(len(upid), bool); m[p] = True; m = m[pc]
        if task == "forecast": m &= meta["fc_keep"].to_numpy()
        return np.flatnonzero(m)
    It, Iv = rows(tr_pat), rows(va_pat)
    dtr = lgb.Dataset(X[It], T[It], free_raw_data=True); dva = lgb.Dataset(X[Iv], T[Iv], reference=dtr)
    for name, c in CFG.items():
        t0 = time.time()
        p = dict(objective="binary", metric=["binary_logloss", "auc", "average_precision"], first_metric_only=True, bagging_fraction=.8,
                 bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=10, **c)
        b = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
        pv = b.predict(X[Iv], num_iteration=b.best_iteration)
        out[f"{task}/{name}"] = {"val_AUROC": float(AUC(T[Iv], pv)), "val_AUPRC": float(AP(T[Iv], pv)), "iters": b.best_iteration, "s": round(time.time()-t0)}
        print(task, name, out[f"{task}/{name}"], flush=True)
json.dump(out, open(f"{R}/results/tuning_val_only.json", "w"), indent=2)
