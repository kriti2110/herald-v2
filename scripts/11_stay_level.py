"""Retrospective STAY-LEVEL detection: does this stay ever become septic? (label = ever SepsisLabel=1)
Inputs: per-stay aggregates (max/min/mean/last) of the audited causal hourly features over the whole stay.
Primary variant EXCLUDES stay length, ICULOS, HospAdmTime, unit and all measurement-timing features;
'with_length' (adds them) is reported only as an ablation. Hyper-parameters chosen on VALIDATION patients
(fold-0 training portion) only; test folds touched once. Patient-level CV + hospital transfer; no resampling."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); info = json.load(open(f"{R}/cache/columns3.json")); cols, groups = info["columns"], info["groups"]
X = np.load(f"{R}/cache/X3.npy", mmap_mode="r"); pc, upid = pd.factorize(meta[PID]); n = len(upid)
folds = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]; fold_of, y, hosp = folds.fold.to_numpy(), folds.ever_septic.to_numpy().astype(int), folds.hospital.to_numpy()
prim = [c for g in ("value", "roll", "composite", "static") for c in groups[g] if not c.endswith("_rate")]
extra = [c for g in ("timing", "care") for c in groups[g]]
def agg(names):
    out = {}
    idx = [cols.index(c) for c in names]
    for s in range(0, len(idx), 60):
        sub = pd.DataFrame(np.asarray(X[:, idx[s:s + 60]]), columns=names[s:s + 60]); gb = sub.groupby(pc)
        for nm, fn in (("max", gb.max()), ("min", gb.min()), ("mean", gb.mean()), ("last", gb.last())):
            fn.columns = [f"{c}__{nm}" for c in sub.columns]; out.update({c: fn[c].to_numpy(np.float32) for c in fn.columns})
    return pd.DataFrame(out)
A1 = agg(prim); A2 = agg(extra); A2["n_hours"] = np.bincount(pc).astype(np.float32)
print("stay-level matrices", A1.shape, A2.shape, flush=True)
BASE = dict(objective="binary", metric=["binary_logloss", "auc"], bagging_fraction=.8, bagging_freq=1, seed=0, verbose=-1, num_threads=10)
GRID = {"s1": dict(learning_rate=.03, num_leaves=7, min_child_samples=50, feature_fraction=.3, lambda_l2=10),
        "s2": dict(learning_rate=.03, num_leaves=15, min_child_samples=50, feature_fraction=.3, lambda_l2=10),
        "s3": dict(learning_rate=.03, num_leaves=15, min_child_samples=100, feature_fraction=.2, lambda_l2=30, extra_trees=True)}
def fit_pred(A, tr, te, cfg, seed=0):
    a, v = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=seed)
    d = lgb.Dataset(A.iloc[a], y[a]); b = lgb.train({**BASE, **cfg, "seed": seed}, d, 3000, valid_sets=[lgb.Dataset(A.iloc[v], y[v], reference=d)], callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    return b.predict(A.iloc[te], num_iteration=b.best_iteration, raw_score=True), b.best_iteration, a, v
# ---- validation-only config selection (fold-0 training portion) ----
tr0 = np.flatnonzero(fold_of != 0); a0, v0 = train_test_split(tr0, test_size=0.25, stratify=y[tr0], random_state=0)
val = {}
for k, c in GRID.items():
    z, it, _, _ = fit_pred(A1, a0, v0, c); val[k] = {"val_AUROC": float(AUC(y[v0], z)), "val_AUPRC": float(AP(y[v0], z)), "iters": it}; print("val", k, val[k], flush=True)
best = max(val, key=lambda k: val[k]["val_AUROC"]); print("chosen config:", best, flush=True)
def boot(yy, zz, reps=1000, seed=0):
    r = np.random.default_rng(seed); au, ap = [], []
    for _ in range(reps):
        i = r.integers(0, len(yy), len(yy)); au.append(AUC(yy[i], zz[i])); ap.append(AP(yy[i], zz[i]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"AUROC_CI95": q(au), "AUPRC_CI95": q(ap)}
def summarize(yy, zz): return {"AUROC": float(AUC(yy, zz)), "AUPRC": float(AP(yy, zz)), "prevalence": float(yy.mean()), "n": int(len(yy)), **boot(yy, zz)}
res = {"validation_only_config_selection": val, "chosen": best}
for vname, A in (("primary_no_length", A1), ("ablation_with_length_and_care", pd.concat([A1, A2], axis=1))):
    oof = np.zeros(n); pf = []
    for k in range(5):
        te = np.flatnonzero(fold_of == k); tr = np.flatnonzero(fold_of != k); z, it, _, _ = fit_pred(A, tr, te, GRID[best]); oof[te] = z
        pf.append((float(AUC(y[te], z)), float(AP(y[te], z)))); print(vname, "fold", k, pf[-1], flush=True)
    res[vname] = {"cv_pooled": summarize(y, oof), "cv_fold_AUROC_mean": float(np.mean([p[0] for p in pf])), "cv_fold_AUROC_sd": float(np.std([p[0] for p in pf], ddof=1)),
                  "cv_fold_AUPRC_mean": float(np.mean([p[1] for p in pf]))}
    tz, ty = [], []
    for a, b in (("A", "B"), ("B", "A")):
        tr = np.flatnonzero(hosp == a); te = np.flatnonzero(hosp == b); z, it, _, _ = fit_pred(A, tr, te, GRID[best]); tz.append(z); ty.append(y[te])
        res[vname][f"transfer_{a}_to_{b}"] = {"AUROC": float(AUC(y[te], z)), "AUPRC": float(AP(y[te], z))}
    res[vname]["transfer_pooled"] = summarize(np.concatenate(ty), np.concatenate(tz))
    print(vname, json.dumps({"cv": res[vname]["cv_pooled"], "transfer": res[vname]["transfer_pooled"]})[:400], flush=True)
json.dump(res, open(f"{R}/results/stay_level.json", "w"), indent=2); print("DONE")
