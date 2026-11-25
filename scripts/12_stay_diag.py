"""Diagnostics for the stay-level model: is the score just stay length? + like-for-like balanced-patient view."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open("scripts/11_stay_level.py").read().split("# ---- validation-only config selection")[0]; exec(src)
n_h = A2["n_hours"].to_numpy()
out = {"AUROC_from_stay_length_alone": float(AUC(y, n_h)), "median_hours_septic": float(np.median(n_h[y == 1])), "median_hours_nonseptic": float(np.median(n_h[y == 0]))}
# primary model OOF, then performance WITHIN length strata (if score were only length, AUROC inside strata ~0.5)
cfg = GRID["s2"]; oof = np.zeros(n)
for k in range(5):
    te = np.flatnonzero(fold_of == k); tr = np.flatnonzero(fold_of != k); oof[te] = fit_pred(A1, tr, te, cfg)[0]
out["primary_CV_AUROC"] = float(AUC(y, oof)); out["primary_CV_AUPRC"] = float(AP(y, oof))
q = np.quantile(n_h, [0, .25, .5, .75, 1.0]); strata = {}
for i in range(4):
    m = (n_h >= q[i]) & (n_h <= q[i + 1]) if i == 3 else (n_h >= q[i]) & (n_h < q[i + 1])
    strata[f"hours[{q[i]:.0f},{q[i+1]:.0f}]"] = {"n": int(m.sum()), "septic": int(y[m].sum()), "AUROC": float(AUC(y[m], oof[m])), "AUPRC": float(AP(y[m], oof[m]))}
out["AUROC_within_length_quartiles"] = strata
rng = np.random.default_rng(0); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); a, b = [], []
for _ in range(200):
    i = np.r_[rng.choice(pos, 432, replace=False), rng.choice(neg, 432, replace=False)]; a.append(AUC(y[i], oof[i])); b.append(AP(y[i], oof[i]))
out["balanced_432_432_view"] = {"AUROC": float(np.mean(a)), "AUPRC": float(np.mean(b))}
json.dump(out, open("results/stay_level_diagnostics.json", "w"), indent=2); print(json.dumps(out, indent=2))
