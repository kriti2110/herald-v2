"""Round-3 ideas, judged on VALIDATION patients only (fold-0 training portion). Test folds untouched.
Adoption rule (fixed beforehand): adopt if val AUROC improves >= +0.001 and val AUPRC does not drop > 3% relative."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID, forecast_rows
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); folds = pd.read_csv(f"{R}/results/patient_folds.csv")
pc, upid = pd.factorize(meta[PID]); f = folds.set_index(PID).loc[upid]; fold_of = f.fold.to_numpy(); ever = f.ever_septic.to_numpy()
train_pat = np.flatnonzero(fold_of != 0)
tr_i, va_i = train_test_split(np.arange(len(train_pat)), test_size=0.25, stratify=ever[train_pat], random_state=0)
tr_pat, va_pat = train_pat[tr_i], train_pat[va_i]; va_pat = train_test_split(va_pat, test_size=0.5, stratify=ever[va_pat], random_state=0)[0]
CFG = {"forecast": {"B": dict(learning_rate=.03, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
                    "C": dict(learning_rate=.03, num_leaves=15, min_child_samples=1000, feature_fraction=.3, lambda_l2=10),
                    "D": dict(learning_rate=.03, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)},
       "detect":   {"D": dict(learning_rate=.03, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True),
                    "B": dict(learning_rate=.03, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
                    "C": dict(learning_rate=.03, num_leaves=15, min_child_samples=1000, feature_fraction=.3, lambda_l2=10)}}
out = {}; Xs = {"v2": np.load(f"{R}/cache/X2.npy"), "v3": np.load(f"{R}/cache/X3.npy")}
_, t3 = forecast_rows(meta, 3); _, t12 = forecast_rows(meta, 12)


def fit(X, task, cfg, y_train, w=None, obj="binary"):
    def rows(p):
        m = np.zeros(len(upid), bool); m[p] = True; m = m[pc]
        if task == "forecast": m &= meta["fc_keep"].to_numpy()
        return np.flatnonzero(m)
    It, Iv = rows(tr_pat), rows(va_pat)
    yt = y_train[It]; yv_hard = (meta["fc_target"] if task == "forecast" else meta["det_target"]).to_numpy()[Iv]
    yv_train = y_train[Iv]
    p = dict(objective=obj, metric=["binary_logloss", "auc"] if obj == "binary" else ["cross_entropy", "auc"], bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=10, **cfg)
    dtr = lgb.Dataset(X[It], yt, weight=None if w is None else w[It]); dva = lgb.Dataset(X[Iv], yv_train, reference=dtr)
    b = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    z = b.predict(X[Iv], num_iteration=b.best_iteration, raw_score=True)
    return z, yv_hard, b.best_iteration


def rec(name, z, y, it=None):
    out[name] = {"val_AUROC": float(AUC(y, z)), "val_AUPRC": float(AP(y, z)), "iters": it}; print(name, out[name], flush=True)


for task in ["forecast", "detect"]:
    hard = (meta["fc_target"] if task == "forecast" else meta["det_target"]).to_numpy().astype(np.float64)
    c0 = list(CFG[task].values())[0]; k0 = list(CFG[task].keys())[0]
    z, y, it = fit(Xs["v2"], task, c0, hard); rec(f"{task}/E0_v2_features", z, y, it)
    z1, y, it = fit(Xs["v3"], task, c0, hard); rec(f"{task}/E1_plus_personal_baseline", z1, y, it)
    if task == "forecast":
        soft = (t3.astype(np.float64) + hard + t12.astype(np.float64)) / 3.0
        z, y, it = fit(Xs["v3"], task, c0, soft, obj="cross_entropy"); rec(f"{task}/E2_multihorizon_soft_label", z, y, it)
    w = np.where(hard > 0.5, 3.0, 1.0)
    z, y, it = fit(Xs["v3"], task, c0, hard, w=w); rec(f"{task}/E3_pos_weight3", z, y, it)
    zs = [z1]
    for k, c in list(CFG[task].items())[1:]:
        zk, y, it = fit(Xs["v3"], task, c, hard); rec(f"{task}/cfg_{k}_v3", zk, y, it); zs.append(zk)
    rec(f"{task}/E4_avg_of_3_configs", np.mean(zs, 0), y)
json.dump(out, open(f"{R}/results/round3_val_only.json", "w"), indent=2); print("DONE")
