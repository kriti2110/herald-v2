"""v1 (306) vs v2 (370) causal features, scored on VALIDATION patients only (fold-0 training portion)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); folds = pd.read_csv(f"{R}/results/patient_folds.csv")
pc, upid = pd.factorize(meta[PID]); f = folds.set_index(PID).loc[upid]; fold_of = f.fold.to_numpy(); ever = f.ever_septic.to_numpy()
train_pat = np.flatnonzero(fold_of != 0)
tr_i, va_i = train_test_split(np.arange(len(train_pat)), test_size=0.25, stratify=ever[train_pat], random_state=0)
tr_pat, va_pat = train_pat[tr_i], train_pat[va_i]; va_pat = train_test_split(va_pat, test_size=0.5, stratify=ever[va_pat], random_state=0)[0]
CFG = {"forecast": dict(learning_rate=.03, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
       "detect": dict(learning_rate=.03, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)}
out = {}
for xf in ["X.npy", "X2.npy"]:
    X = np.load(f"{R}/cache/{xf}")
    for task in ["forecast", "detect"]:
        T = meta["fc_target"].to_numpy() if task == "forecast" else meta["det_target"].to_numpy()
        def rows(p):
            m = np.zeros(len(upid), bool); m[p] = True; m = m[pc]
            if task == "forecast": m &= meta["fc_keep"].to_numpy()
            return np.flatnonzero(m)
        It, Iv = rows(tr_pat), rows(va_pat)
        dtr = lgb.Dataset(X[It], T[It]); dva = lgb.Dataset(X[Iv], T[Iv], reference=dtr)
        p = dict(objective="binary", metric=["binary_logloss", "auc", "average_precision"], bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=10, **CFG[task])
        b = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
        pv = b.predict(X[Iv], num_iteration=b.best_iteration)
        out[f"{xf}/{task}"] = {"val_AUROC": float(AUC(T[Iv], pv)), "val_AUPRC": float(AP(T[Iv], pv)), "iters": b.best_iteration}
        print(xf, task, out[f"{xf}/{task}"], flush=True)
    del X
json.dump(out, open(f"{R}/results/features_v1_vs_v2_val_only.json", "w"), indent=2)
