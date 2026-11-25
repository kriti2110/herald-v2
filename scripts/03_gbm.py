"""Causal-feature gradient boosting for BOTH tasks, patient-level evaluation.

  --mode cv        5-fold patient-level stratified CV over all 40,336 stays
  --mode transfer  train on one hospital, test on the other (A->B, B->A)
  --task forecast  early prediction: onset (first positive label) within next 6 h
  --task detect    detection: SepsisLabel at the current hour (all hours)
  --variant        full | no_care (drop ICULOS/HospAdmTime/Unit*) |
                   physio_only (also drop all measurement-timing features)
Within each training portion: patient-level 75/12.5/12.5 fit/val/calib.
Val -> early stopping; calib -> isotonic calibration. Test is touched once.
Test sets are NEVER resampled or balanced.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from hv2.features import PID

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="cv", choices=["cv", "transfer"])
ap.add_argument("--task", required=True, choices=["forecast", "detect"])
ap.add_argument("--variant", default="full", choices=["full", "no_care", "physio_only"])
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

meta = pd.read_parquet(f"{R}/cache/meta.parquet")
info = json.load(open(f"{R}/cache/columns.json"))
cols, groups = info["columns"], info["groups"]
drop = set()
if args.variant in ("no_care", "physio_only"):
    drop |= set(groups["care"])
if args.variant == "physio_only":
    drop |= set(groups["timing"])
use = [i for i, c in enumerate(cols) if c not in drop]
fnames = [cols[i] for i in use]
X = np.load(f"{R}/cache/X.npy", mmap_mode="r")
X = np.ascontiguousarray(X[:, use])
print(f"variant={args.variant} features={len(use)} task={args.task} mode={args.mode}", flush=True)

pc, upid = pd.factorize(meta[PID])
ever = meta.groupby(PID)["SepsisLabel"].max().reindex(upid).to_numpy() > 0.5
hosp = meta.groupby(PID)["hospital"].first().reindex(upid).to_numpy()
fc_keep, fc_t, det_t = meta["fc_keep"].to_numpy(), meta["fc_target"].to_numpy(), meta["det_target"].to_numpy()
T = fc_t if args.task == "forecast" else det_t


def rows(pat_idx):
    member = np.zeros(len(upid), bool); member[pat_idx] = True
    m = member[pc]
    if args.task == "forecast":
        m &= fc_keep
    return np.flatnonzero(m)


def patient_level(idx, y, p):
    d = pd.DataFrame({"pc": pc[idx], "y": y, "p": p}).groupby("pc").agg(y=("y", "max"), p=("p", "max"))
    return d


def balanced_patient_metrics(d, reps=200, seed=0):
    """Like-for-like with SepsisAI's protocol: equal numbers of septic and
    non-septic PATIENTS (a resampled VIEW of the held-out predictions only;
    the headline metrics are never computed on resampled data)."""
    rng = np.random.default_rng(seed)
    pos, neg = d[d.y == 1], d[d.y == 0]
    n = min(len(pos), len(neg), 432)
    a, b = [], []
    for _ in range(reps):
        s = pd.concat([pos.sample(n, random_state=int(rng.integers(1e9))), neg.sample(n, random_state=int(rng.integers(1e9)))])
        a.append(AUC(s.y, s.p)); b.append(AP(s.y, s.p))
    return {"n_per_class": n, "AUROC": float(np.mean(a)), "AUPRC": float(np.mean(b))}


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-12
    e = 0.0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def run(train_pat, test_pat, tag):
    t0 = time.time()
    tr_pat, va_pat = train_test_split(train_pat, test_size=0.25, stratify=ever[train_pat], random_state=args.seed)
    va_pat, ca_pat = train_test_split(va_pat, test_size=0.5, stratify=ever[va_pat], random_state=args.seed)
    I = {k: rows(v) for k, v in dict(fit=tr_pat, val=va_pat, cal=ca_pat, test=test_pat).items()}
    # per-task configs chosen on VALIDATION patients only (scripts/04_tune.py)
    cfg = {"forecast": dict(learning_rate=.03, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
           "detect": dict(learning_rate=.03, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)}[args.task]
    params = dict(objective="binary", metric=["binary_logloss", "auc", "average_precision"], first_metric_only=True, bagging_fraction=0.8,
                  bagging_freq=1, max_bin=127, seed=args.seed, verbose=-1, num_threads=10, **cfg)
    dtr = lgb.Dataset(X[I["fit"]], T[I["fit"]], feature_name=fnames, free_raw_data=True)
    dva = lgb.Dataset(X[I["val"]], T[I["val"]], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=["val"],
                    callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
    pe = lambda k: bst.predict(X[I[k]], num_iteration=bst.best_iteration)
    p_cal, p_te = pe("cal"), pe("test")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p_cal, T[I["cal"]])
    p_te_cal = iso.predict(p_te)
    y = T[I["test"]]
    d = patient_level(I["test"], y, p_te)
    m = {
        "tag": tag, "n_test_patients": int(len(test_pat)), "n_test_rows": int(len(y)), "n_test_pos": int(y.sum()),
        "prevalence": float(y.mean()), "AUROC": float(AUC(y, p_te)), "AUPRC": float(AP(y, p_te)),
        "patient_AUROC": float(AUC(d.y, d.p)), "patient_AUPRC": float(AP(d.y, d.p)),
        "patient_balanced_view": balanced_patient_metrics(d),
        "Brier_calibrated": float(brier_score_loss(y, p_te_cal)), "ECE_calibrated": ece(y, p_te_cal),
        "best_iteration": int(bst.best_iteration), "seconds": round(time.time() - t0),
    }
    imp = pd.Series(bst.feature_importance("gain"), index=fnames).sort_values(ascending=False)
    m["top_features_gain"] = {k: float(v) for k, v in imp.head(15).items()}
    out = pd.DataFrame({PID: upid[pc[I["test"]]], "ICULOS": meta["ICULOS"].to_numpy()[I["test"]], "y": y,
                        "p": p_te, "p_cal": p_te_cal})
    print(json.dumps({k: m[k] for k in ["tag", "AUROC", "AUPRC", "patient_AUROC", "prevalence", "best_iteration", "seconds"]}), flush=True)
    return m, out


name = f"{args.mode}_{args.task}_{args.variant}"
res, preds = [], []
if args.mode == "cv":
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    allp = np.arange(len(upid))
    fold_of = np.zeros(len(upid), int)
    for k, (_, te) in enumerate(skf.split(allp, ever)):
        fold_of[te] = k
    pd.DataFrame({PID: upid, "fold": fold_of, "hospital": hosp, "ever_septic": ever}).to_csv(f"{R}/results/patient_folds.csv", index=False)
    for k in [int(f) for f in args.folds.split(",")]:
        m, o = run(allp[fold_of != k], allp[fold_of == k], f"fold{k}")
        m["fold"] = k; o["fold"] = k; res.append(m); preds.append(o)
else:
    for a, b in [("A", "B"), ("B", "A")]:
        m, o = run(np.flatnonzero(hosp == a), np.flatnonzero(hosp == b), f"train{a}_test{b}")
        o["direction"] = f"{a}->{b}"; res.append(m); preds.append(o)
json.dump(res, open(f"{R}/results/{name}.json", "w"), indent=2)
pd.concat(preds).to_parquet(f"{R}/results/{name}_preds.parquet")
print("saved", name)
