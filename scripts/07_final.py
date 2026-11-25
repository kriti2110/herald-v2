"""HERALD-v2 FINAL model (design fixed before test results were examined):
  Stage 1: LightGBM on 370 causal features. Out-of-fold (4 inner patient folds) logits for training rows;
           test logits = mean of the 4 inner models (same distribution as OOF).
  Stage 2: LightGBM on the causal HISTORY of the stage-1 score (running max, trailing means/max, trends,
           threshold-exceedance counts) + top stage-1 features + ICULOS.
  Same code for both tasks; patient-level CV or hospital transfer; test sets never resampled."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold, train_test_split
from hv2.features import PID

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="cv", choices=["cv", "transfer"]); ap.add_argument("--task", required=True, choices=["forecast", "detect"])
ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--threads", type=int, default=5); ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

meta = pd.read_parquet(f"{R}/cache/meta.parquet"); info = json.load(open(f"{R}/cache/columns2.json")); cols = info["columns"]
X = np.load(f"{R}/cache/X2.npy")
pc, upid = pd.factorize(meta[PID]); folds = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]
fold_of, ever, hosp = folds.fold.to_numpy(), folds.ever_septic.to_numpy(), folds.hospital.to_numpy()
fc_keep = meta.fc_keep.to_numpy(); T = meta.fc_target.to_numpy() if args.task == "forecast" else meta.det_target.to_numpy()
ICU = meta.ICULOS.to_numpy()
S1 = {"forecast": dict(learning_rate=.05, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
      "detect": dict(learning_rate=.05, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)}[args.task]
BASE = dict(objective="binary", metric=["binary_logloss", "auc"], bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=args.threads)
ES = lambda: lgb.early_stopping(100, first_metric_only=True, verbose=False)


def rows_of(pats):
    m = np.zeros(len(upid), bool); m[pats] = True; m = m[pc]
    if args.task == "forecast": m &= fc_keep
    return np.flatnonzero(m)


def hist_feats(codes, z, thr):
    """Causal history of a score series. `codes` = contiguous patient codes (rows in time order)."""
    s = pd.Series(z); gb = s.groupby(codes); f = {"z": z, "z_cummax": gb.cummax().to_numpy()}
    for w in (3, 6, 12, 24): f[f"z_mean{w}"] = gb.rolling(w, min_periods=1).mean().to_numpy()
    for w in (6, 24): f[f"z_max{w}"] = gb.rolling(w, min_periods=1).max().to_numpy()
    f["dz3"] = (s - gb.shift(3)).to_numpy(); f["dz6"] = (s - gb.shift(6)).to_numpy()
    for nm, t in thr.items():
        fl = pd.Series((z > t).astype(np.float32)).groupby(codes)
        f[f"n_{nm}_6"] = fl.rolling(6, min_periods=1).sum().to_numpy(); f[f"n_{nm}_24"] = fl.rolling(24, min_periods=1).sum().to_numpy()
    return pd.DataFrame(f)


def selftest():
    """Stage-2 history features must not change when later rows are appended."""
    rng = np.random.default_rng(0); z = rng.normal(size=300); codes = np.repeat(np.arange(3), 100); thr = {"a": 0.5}
    full = hist_feats(codes, z, thr)
    for t in (10, 40, 70):
        pre = hist_feats(codes[: 100 + t + 1], z[: 100 + t + 1], thr)
        assert np.allclose(pre.iloc[-1].to_numpy(), full.iloc[100 + t].to_numpy(), equal_nan=True), "history features leak future"
    print("stage-2 history self-test PASS", flush=True)


def fit(Xa, ya, Xv, yv, params, names=None):
    d = lgb.Dataset(Xa, ya, feature_name=names or "auto"); v = lgb.Dataset(Xv, yv, reference=d)
    return lgb.train({**BASE, **params}, d, 3000, valid_sets=[v], callbacks=[ES()])


def run(P, Te, tag):
    t0 = time.time()
    if args.smoke: P = np.random.default_rng(0).choice(P, 3000, replace=False); Te = Te[:600]
    iT = rows_of(Te); iP = rows_of(P); zP = np.zeros(len(iP)); zT = np.zeros(len(iT)); gain = np.zeros(len(cols))
    pos_in_P = {r: i for i, r in enumerate(iP)}
    inner = StratifiedKFold(4, shuffle=True, random_state=1)
    for j, (a, b) in enumerate(inner.split(P, ever[P])):
        a_tr, a_va = train_test_split(P[a], test_size=0.12, stratify=ever[P[a]], random_state=j)
        Ia, Iv, Ib = rows_of(a_tr), rows_of(a_va), rows_of(P[b])
        m = fit(X[Ia], T[Ia], X[Iv], T[Iv], S1)
        zb = m.predict(X[Ib], num_iteration=m.best_iteration, raw_score=True)
        zP[np.searchsorted(iP, Ib)] = zb
        zT += m.predict(X[iT], num_iteration=m.best_iteration, raw_score=True) / 4; gain += m.feature_importance("gain") / 4
        print(f"[{tag}] stage1 inner {j} iters {m.best_iteration} {time.time()-t0:.0f}s", flush=True)
    top = [cols[i] for i in np.argsort(-gain)[:25]]; topi = [cols.index(c) for c in top]
    thr = {"q90": float(np.quantile(zP, .90)), "q99": float(np.quantile(zP, .99))}
    FP = hist_feats(pc[iP], zP, thr); FT = hist_feats(pc[iT], zT, thr)
    for F_, ii in ((FP, iP), (FT, iT)):
        F_["ICULOS"] = ICU[ii]
        for c, k in zip(top, topi): F_[c] = X[ii, k]
    p2, v2 = train_test_split(P, test_size=0.15, stratify=ever[P], random_state=7)
    mp = {r: i for i, r in enumerate(iP)}
    ia = np.searchsorted(iP, rows_of(p2)); iv = np.searchsorted(iP, rows_of(v2))
    S2 = dict(learning_rate=.05, num_leaves=15, min_child_samples=500, feature_fraction=.5, lambda_l2=10)
    m2 = fit(FP.iloc[ia], T[iP][ia], FP.iloc[iv], T[iP][iv], S2)
    z2 = m2.predict(FT, num_iteration=m2.best_iteration, raw_score=True)
    y = T[iT]; res = {"tag": tag, "n_test_patients": int(len(Te)), "prevalence": float(y.mean()),
                      "stage1_AUROC": float(AUC(y, zT)), "stage1_AUPRC": float(AP(y, zT)),
                      "stage2_AUROC": float(AUC(y, z2)), "stage2_AUPRC": float(AP(y, z2)), "stage2_iters": int(m2.best_iteration), "seconds": round(time.time() - t0)}
    print(json.dumps(res), flush=True)
    return res, pd.DataFrame({PID: upid[pc[iT]], "ICULOS": ICU[iT], "y": y, "z1": zT, "z2": z2})


selftest()
name = f"final_{args.mode}_{args.task}" + ("_smoke" if args.smoke else ""); res, preds = [], []
allp = np.arange(len(upid))
if args.mode == "cv":
    for k in [int(f) for f in args.folds.split(",")]:
        r, o = run(allp[fold_of != k], allp[fold_of == k], f"fold{k}"); o["fold"] = k; res.append(r); preds.append(o)
        if args.smoke: break
else:
    for a, b in [("A", "B"), ("B", "A")]:
        r, o = run(np.flatnonzero(hosp == a), np.flatnonzero(hosp == b), f"train{a}_test{b}"); o["direction"] = f"{a}->{b}"; res.append(r); preds.append(o)
        if args.smoke: break
json.dump(res, open(f"{R}/results/{name}.json", "w"), indent=2); pd.concat(preds).to_parquet(f"{R}/results/{name}_preds.parquet"); print("saved", name)
