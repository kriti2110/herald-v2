"""Subgroup analysis (age band, sex, hospital), decision-curve analysis and alert-burden metrics.
Reads only saved out-of-fold predictions (final two-stage score z2). Nothing is retrained."""
import json, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
R = "/Users/kritiraj/Desktop/HERALD-v2"
rng = np.random.default_rng(42)
raw = pd.read_parquet(f"{R}/cache/raw_long.parquet", columns=["patient_id", "Age", "Gender"])
pat = raw.groupby("patient_id").agg(Age=("Age", "first"), Gender=("Gender", "first"))
hosp = pd.read_csv(f"{R}/results/patient_folds.csv").set_index("patient_id")["hospital"]
pat["hospital"] = hosp
pat["ageband"] = pd.cut(pat.Age, [0, 45, 65, 80, 200], right=False, labels=["<45", "45-64", "65-79", ">=80"])
pat["sex"] = pat.Gender.map({0: "Female", 1: "Male"})

def boot_auc(y, p, pid, reps=300):
    codes, uniq = pd.factorize(pid); n = len(uniq)
    o = np.argsort(p, kind="stable"); y, codes = y[o], codes[o]; au = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[codes]
        wp, wn = w * y, w * (1 - y); cn = np.cumsum(wn) - wn
        den = wp.sum() * wn.sum()
        if den > 0: au.append((wp * cn).sum() / den)
    return np.percentile(au, [2.5, 97.5])

def boot_ap(y, p, pid, reps=300):
    """Patient-clustered bootstrap of average precision (separate RNG so AUROC intervals are unchanged)."""
    rg = np.random.default_rng(43); codes, uniq = pd.factorize(pid); n = len(uniq)
    o = np.argsort(-p, kind="stable"); y = y[o].astype(float); codes = codes[o]; vals = []
    for _ in range(reps):
        w = np.bincount(rg.integers(0, n, n), minlength=n)[codes].astype(float); wp = w * y; tp = np.cumsum(wp); fp = np.cumsum(w * (1 - y))
        den = wp.sum()
        if den > 0: vals.append((wp * tp / np.maximum(tp + fp, 1e-12)).sum() / den)
    return np.percentile(vals, [2.5, 97.5])

def calibrate(d):   # cross-fold isotonic: fit on the other folds' OOF scores, apply to held-out fold
    p = np.zeros(len(d))
    for f in sorted(d.fold.unique()):
        tr, te = d.fold != f, d.fold == f
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d.z2[tr], d.y[tr]); p[te.to_numpy()] = ir.predict(d.z2[te])
    return p

out = {"subgroups": [], "dca": {}, "burden": {}}
for task, f in [("Early 6 h", "final_cv_forecast_preds"), ("Detection", "final_cv_detect_preds")]:
    d = pd.read_parquet(f"{R}/results/{f}.parquet").merge(pat, left_on="patient_id", right_index=True, how="left")
    assert d.Age.notna().all() and d.hospital.notna().all()
    for gname, col in [("Age (years)", "ageband"), ("Sex", "sex"), ("Hospital", "hospital")]:
        for lv in [x for x in d[col].cat.categories] if str(d[col].dtype) == "category" else sorted(d[col].dropna().unique()):
            s = d[d[col] == lv]; y = s.y.to_numpy(); p = s.z2.to_numpy()
            ci = boot_auc(y, p, s.patient_id.to_numpy())
            cap = boot_ap(y, p, s.patient_id.to_numpy())
            out["subgroups"].append(dict(task=task, group=gname, level=str(lv), n_patients=int(s.patient_id.nunique()), n_rows=int(len(s)),
                prevalence=float(y.mean()), auroc=float(AUC(y, p)), auroc_lo=float(ci[0]), auroc_hi=float(ci[1]),
                auprc=float(AP(y, p)), auprc_lo=float(cap[0]), auprc_hi=float(cap[1]), lift=float(AP(y, p) / y.mean())))
    # decision curve (row level, cross-fold isotonic-calibrated risk)
    d["p"] = calibrate(d); y = d.y.to_numpy(); p = d.p.to_numpy(); N = len(d); prev = y.mean()
    th = np.round(np.arange(0.005, 0.1501, 0.005), 3); nb, nball = [], []
    for t in th:
        a = p >= t; tp = (a & (y == 1)).sum(); fp = (a & (y == 0)).sum(); w = t / (1 - t)
        nb.append(tp / N - fp / N * w); nball.append(prev - (1 - prev) * w)
    out["dca"][task] = dict(thresholds=th.tolist(), net_benefit=nb, treat_all=nball, prevalence=float(prev))
    # alert burden at 90 / 95 % row specificity
    hours = N; days = hours / 24.0
    for spec in (0.90, 0.95):
        thr = np.quantile(p[y == 0], spec) if False else np.quantile(d.z2[y == 0], spec)
        a = d.z2.to_numpy() > thr; tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
        # lead time: for septic stays, hours between first alert and first positive-label row (alert strictly earlier only)
        lead = []; alerted = 0; ns = 0
        for pid, g in d.groupby("patient_id", sort=False):
            if g.y.max() == 0 and task == "Detection": pass
            gy = g.y.to_numpy(); gz = g.z2.to_numpy() > thr; it = g.ICULOS.to_numpy()
            if task == "Early 6 h":
                # forecast rows only exist before label time; septic stay: positive rows are the 6 h before tau
                if gy.max() == 0: continue
                ns += 1; tau = it[gy == 1].max() + 1  # first label hour = last positive-target hour + 1
            else:
                if gy.max() == 0: continue
                ns += 1; tau = it[gy == 1].min()
            ah = it[gz & (it < tau)]
            if len(ah): alerted += 1; lead.append(float(tau - ah.min()))
        out["burden"][f"{task}|spec{int(spec*100)}"] = dict(false_alarms_per_patient_day=fp / days, alerts_per_tp=(tp + fp) / max(tp, 1),
            sens=tp / y.sum(), septic_stays=ns, septic_alerted_before_label=alerted, lead_median=float(np.median(lead)) if lead else None,
            lead_q1=float(np.percentile(lead, 25)) if lead else None, lead_q3=float(np.percentile(lead, 75)) if lead else None)
json.dump(out, open(f"{R}/results/subgroups_dca_burden.json", "w"), indent=1)
print(pd.DataFrame(out["subgroups"]).round(3).to_string()); print(json.dumps(out["burden"], indent=1))
