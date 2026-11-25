"""Aggregate every experiment into final, exact numbers (results/final_metrics.json + RESULTS.md)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, brier_score_loss
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PID = "patient_id"; rng = np.random.default_rng(0)
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def fast_boot(y, p, pidx, reps=1000):
    """Patient-clustered bootstrap. Sorts once; each replicate re-weights whole patients."""
    n = pidx.max() + 1
    o = np.argsort(p, kind="stable"); y, p, pidx = y[o], p[o], pidx[o]
    au, ap = [], []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[pidx].astype(np.float64)
        wp, wn = w * y, w * (1 - y)
        cn = np.cumsum(wn) - wn
        au.append((wp * cn).sum() / (wp.sum() * wn.sum()))
        wp_d, w_d = wp[::-1], w[::-1]                     # descending score
        prec = np.cumsum(wp_d) / np.maximum(np.cumsum(w_d), 1e-12)
        ap.append((wp_d * prec).sum() / wp.sum())
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"AUROC_CI95": q(au), "AUPRC_CI95": q(ap)}


def metrics(df, s, y="y", with_ci=False):
    yy, ss = df[y].to_numpy(), df[s].to_numpy()
    pat = df.groupby(PID).agg(y=(y, "max"), s=(s, "max"))
    m = {"AUROC": float(AUC(yy, ss)), "AUPRC": float(AP(yy, ss)), "patient_AUROC": float(AUC(pat.y, pat.s)),
         "patient_AUPRC": float(AP(pat.y, pat.s)), "prevalence": float(yy.mean()), "n_rows": int(len(yy)),
         "n_pos_rows": int(yy.sum()), "n_patients": int(len(pat))}
    if with_ci:
        m.update(fast_boot(yy.astype(np.float64), ss, pd.factorize(df[PID])[0]))
    return m


def balanced_patient(df, s, y="y", reps=200, n=432):
    pat = df.groupby(PID).agg(y=(y, "max"), s=(s, "max")); pos, neg = pat[pat.y == 1], pat[pat.y == 0]
    a, b = [], []
    for _ in range(reps):
        t = pd.concat([pos.sample(n, random_state=int(rng.integers(1e9))), neg.sample(n, random_state=int(rng.integers(1e9)))])
        a.append(AUC(t.y, t.s)); b.append(AP(t.y, t.s))
    return {"n_per_class": n, "AUROC": float(np.mean(a)), "AUPRC": float(np.mean(b))}


def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-12; e = 0.0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def load(mode, task, variant):
    f = f"{R}/results/{mode}_{task}_{variant}_preds.parquet"
    return pd.read_parquet(f) if os.path.exists(f) else None


def blend(mode, task, group):
    g = load(mode, task, "full"); t = pd.read_parquet(f"{R}/results/{mode}_tcn_preds.parquet")
    zc, yc = ("z_fc", "y_fc") if task == "forecast" else ("z_det", "y_det")
    if task == "forecast": t = t[t.fc_keep]
    d = g.merge(t[[PID, "ICULOS", zc]], on=[PID, "ICULOS"], how="inner"); assert len(d) == len(g), (len(d), len(g))
    d["z_gbm"] = lg(d["p"].to_numpy()); d["z_tcn"] = d[zc]; d["z_ens"] = 0.5 * d["z_gbm"] + 0.5 * d["z_tcn"]
    return d


def calibrate_crossfold(d, group):
    """Isotonic calibration of the ensemble: fit on the OTHER folds' out-of-fold scores."""
    out = np.zeros(len(d))
    for k in d[group].unique():
        te = d[group] == k; iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(d.loc[~te, "z_ens"], d.loc[~te, "y"])
        out[te.to_numpy()] = iso.predict(d.loc[te, "z_ens"])
    return out



final = {}
for task in ["forecast", "detect"]:
    for mode, group in [("cv", "fold"), ("transfer", "direction")]:
        d = pd.read_parquet(f"{R}/results/final_{mode}_{task}_preds.parquet"); key = f"{mode}/{task}"; final[key] = {}
        for name, col in [("stage1", "z1"), ("FINAL_stage2", "z2")]:
            res = {"pooled": metrics(d, col, with_ci=True), "per_group": {str(k): metrics(g, col) for k, g in d.groupby(group)}}
            for mm in ["AUROC", "AUPRC", "patient_AUROC", "patient_AUPRC"]:
                v = [x[mm] for x in res["per_group"].values()]; res[f"{mm}_mean"] = float(np.mean(v)); res[f"{mm}_sd"] = float(np.std(v, ddof=1))
            res["balanced_patient_view"] = balanced_patient(d, col); final[key][name] = res
        if mode == "cv":
            out = np.zeros(len(d))
            for k in d[group].unique():
                te = (d[group] == k).to_numpy(); iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(d.loc[~te, "z2"], d.loc[~te, "y"]); out[te] = iso.predict(d.loc[te, "z2"])
            final[key]["FINAL_stage2"]["calibration_crossfold_isotonic"] = {"Brier": float(brier_score_loss(d.y, out)), "ECE": ece(d.y.to_numpy(), out), "Brier_constant_prevalence": float(brier_score_loss(d.y, np.full(len(d), d.y.mean())))}
json.dump(final, open(f"{R}/results/final_twostage_metrics.json", "w"), indent=2)
for k, v in final.items():
    print("\n==", k)
    for n, m in v.items():
        p = m["pooled"]; b = m["balanced_patient_view"]
        print(f"{n:13s} AUROC {p['AUROC']:.4f} [{p['AUROC_CI95'][0]:.4f},{p['AUROC_CI95'][1]:.4f}]  AUPRC {p['AUPRC']:.4f} [{p['AUPRC_CI95'][0]:.4f},{p['AUPRC_CI95'][1]:.4f}]  patAUROC {p['patient_AUROC']:.4f} patAUPRC {p['patient_AUPRC']:.4f} | folds {m['AUROC_mean']:.4f}+-{m['AUROC_sd']:.4f} / {m['AUPRC_mean']:.4f}+-{m['AUPRC_sd']:.4f} | balanced-patient {b['AUROC']:.4f}/{b['AUPRC']:.4f}")
    if "calibration_crossfold_isotonic" in v["FINAL_stage2"]: print("  calibration", v["FINAL_stage2"]["calibration_crossfold_isotonic"])
    if k.startswith("transfer"):
        for g, x in v["FINAL_stage2"]["per_group"].items(): print("   ", g, round(x["AUROC"], 4), round(x["AUPRC"], 4))
