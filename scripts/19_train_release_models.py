"""Train the final two-stage model on ALL patients and save it (release artifact). Same data, hyper-parameters and inner-split
logic as scripts/07_final.py, with the outer test fold removed (P = every patient). No test predictions are made, so nothing
here changes the reported cross-validated results.   python scripts/19_train_release_models.py --task forecast|detect"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from hv2.features import PID
from hv2.twostage import hist_feats
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["forecast", "detect"]); ap.add_argument("--threads", type=int, default=5); args = ap.parse_args()
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); info = json.load(open(f"{R}/cache/columns2.json")); cols = info["columns"]; X = np.load(f"{R}/cache/X2.npy")
pc, upid = pd.factorize(meta[PID]); folds = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]; ever = folds.ever_septic.to_numpy()
fc_keep = meta.fc_keep.to_numpy(); T = meta.fc_target.to_numpy() if args.task == "forecast" else meta.det_target.to_numpy(); ICU = meta.ICULOS.to_numpy()
S1 = {"forecast": dict(learning_rate=.05, num_leaves=31, min_child_samples=500, feature_fraction=.3, lambda_l2=10),
      "detect": dict(learning_rate=.05, num_leaves=31, min_child_samples=2000, feature_fraction=.2, lambda_l2=30, extra_trees=True)}[args.task]
S2 = dict(learning_rate=.05, num_leaves=15, min_child_samples=500, feature_fraction=.5, lambda_l2=10)
BASE = dict(objective="binary", metric=["binary_logloss", "auc"], bagging_fraction=.8, bagging_freq=1, max_bin=127, seed=0, verbose=-1, num_threads=args.threads)
ES = lambda: lgb.early_stopping(100, first_metric_only=True, verbose=False)
def rows_of(pats):
    m = np.zeros(len(upid), bool); m[pats] = True; m = m[pc]
    if args.task == "forecast": m &= fc_keep
    return np.flatnonzero(m)
def fit(Xa, ya, Xv, yv, params, names=None):
    d = lgb.Dataset(Xa, ya, feature_name=names or "auto"); v = lgb.Dataset(Xv, yv, reference=d); return lgb.train({**BASE, **params}, d, 3000, valid_sets=[v], callbacks=[ES()])
out = f"{R}/models/{args.task}"; os.makedirs(out, exist_ok=True); t0 = time.time()
P = np.arange(len(upid)); iP = rows_of(P); zP = np.zeros(len(iP)); gain = np.zeros(len(cols)); iters = []
for j, (a, b) in enumerate(StratifiedKFold(4, shuffle=True, random_state=1).split(P, ever[P])):
    a_tr, a_va = train_test_split(P[a], test_size=0.12, stratify=ever[P[a]], random_state=j)
    Ia, Iv, Ib = rows_of(a_tr), rows_of(a_va), rows_of(P[b]); m = fit(X[Ia], T[Ia], X[Iv], T[Iv], S1)
    zP[np.searchsorted(iP, Ib)] = m.predict(X[Ib], num_iteration=m.best_iteration, raw_score=True); gain += m.feature_importance("gain") / 4
    m.save_model(f"{out}/stage1_inner{j}.txt", num_iteration=m.best_iteration); iters.append(int(m.best_iteration)); print(f"[{args.task}] stage1 inner {j} iters {m.best_iteration} {time.time()-t0:.0f}s", flush=True)
top = [cols[i] for i in np.argsort(-gain)[:25]]; topi = [cols.index(c) for c in top]
thr = {"q90": float(np.quantile(zP, .90)), "q99": float(np.quantile(zP, .99))}
FP = hist_feats(pc[iP], zP, thr); FP["ICULOS"] = ICU[iP]
for c, k in zip(top, topi): FP[c] = X[iP, k]
p2, v2 = train_test_split(P, test_size=0.15, stratify=ever[P], random_state=7)
ia = np.searchsorted(iP, rows_of(p2)); iv = np.searchsorted(iP, rows_of(v2)); m2 = fit(FP.iloc[ia], T[iP][ia], FP.iloc[iv], T[iP][iv], S2)
m2.save_model(f"{out}/stage2.txt", num_iteration=m2.best_iteration)
json.dump(dict(task=args.task, columns=cols, top25=top, thresholds=thr, stage1_params={**BASE, **S1, "num_threads": "n/a"}, stage2_params={**BASE, **S2, "num_threads": "n/a"},
               stage1_best_iterations=iters, stage2_best_iteration=int(m2.best_iteration), n_patients=int(len(P)), n_rows=int(len(iP)), seeds=dict(lightgbm=0, inner_kfold=1, inner_val="j (0..3)", stage2_split=7),
               note="Trained on all 40,336 patients for release; reported results come from cross-validation (results/)."), open(f"{out}/model_info.json", "w"), indent=1)
print(f"[{args.task}] done {time.time()-t0:.0f}s stage2 iters {m2.best_iteration}", flush=True)
