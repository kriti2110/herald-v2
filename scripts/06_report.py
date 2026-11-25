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
        if load(mode, task, "full") is None or not os.path.exists(f"{R}/results/{mode}_tcn_preds.parquet"):
            print("skip", mode, task); continue
        d = blend(mode, task, group); key = f"{mode}/{task}"; final[key] = {}
        for name, col in [("GBM", "z_gbm"), ("TCN", "z_tcn"), ("ENSEMBLE", "z_ens")]:
            res = {"pooled": metrics(d, col, with_ci=True), "per_group": {}}
            for k, g in d.groupby(group):
                res["per_group"][str(k)] = metrics(g, col)
            for mm in ["AUROC", "AUPRC", "patient_AUROC"]:
                v = [x[mm] for x in res["per_group"].values()]; res[f"{mm}_mean"] = float(np.mean(v)); res[f"{mm}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            res["balanced_patient_view"] = balanced_patient(d, col)
            final[key][name] = res
        pcal = calibrate_crossfold(d, group) if mode == "cv" else None
        if pcal is not None:
            final[key]["ENSEMBLE"]["calibration_crossfold_isotonic"] = {"Brier": float(brier_score_loss(d.y, pcal)), "ECE": ece(d.y.to_numpy(), pcal),
                                                                        "Brier_of_constant_prevalence": float(brier_score_loss(d.y, np.full(len(d), d.y.mean())))}
        print(key, {n: (round(final[key][n]["pooled"]["AUROC"], 4), round(final[key][n]["pooled"]["AUPRC"], 4)) for n in final[key]}, flush=True)
    # ablations (GBM only)
    for mode, group in [("cv", "fold"), ("transfer", "direction")]:
        for v in ["no_care", "physio_only"]:
            a = load(mode, task, v)
            if a is not None:
                final.setdefault(f"ablation/{mode}/{task}", {})[v] = {"pooled": metrics(a.assign(s=lg(a.p)), "s"), "per_group": {str(k): metrics(g.assign(s=lg(g.p)), "s") for k, g in a.groupby(group)}}
        f = load(mode, task, "full")
        if f is not None:
            final.setdefault(f"ablation/{mode}/{task}", {})["full"] = {"pooled": metrics(f.assign(s=lg(f.p)), "s"), "per_group": {str(k): metrics(g.assign(s=lg(g.p)), "s") for k, g in f.groupby(group)}}
json.dump(final, open(f"{R}/results/final_metrics.json", "w"), indent=2)
print("wrote final_metrics.json")
