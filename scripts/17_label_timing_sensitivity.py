"""Label-timing sensitivity (evaluation only, nothing is retrained).
The early-prediction scores of the final model are re-scored against labels whose onset time is shifted EARLIER by delta hours:
tau' = tau+ + delta (delta <= 0), target' = 1{tau' - t <= 6, t < tau'}; rows with t >= tau' of septic stays are dropped.
(Later onsets cannot be evaluated because forecast rows after tau+ were never scored.) delta = 0 reproduces the primary result."""
import json, os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
R = os.environ.get("HERALD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rng = np.random.default_rng(42)
meta = pd.read_parquet(f"{R}/cache/meta.parquet", columns=["patient_id", "ICULOS", "SepsisLabel"])
tau = meta[meta.SepsisLabel == 1].groupby("patient_id").ICULOS.min().rename("tau")
d = pd.read_parquet(f"{R}/results/final_cv_forecast_preds.parquet").merge(tau, left_on="patient_id", right_index=True, how="left")
d["tau"] = d["tau"].fillna(np.inf)
# check: reconstruct the primary target
chk = ((d.tau - d.ICULOS <= 6) & (d.ICULOS < d.tau)).astype(int)
assert (chk.to_numpy() == d.y.to_numpy()).all(), "target reconstruction mismatch"
def boot_auc(y, p, pid, reps=300):
    codes, uniq = pd.factorize(pid); n = len(uniq); o = np.argsort(p, kind="stable"); y, codes = y[o], codes[o]; au = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[codes]; wp, wn = w * y, w * (1 - y); cn = np.cumsum(wn) - wn; den = wp.sum() * wn.sum()
        if den > 0: au.append((wp * cn).sum() / den)
    return np.percentile(au, [2.5, 97.5])
out = []
for delta in (0, -1, -3, -6):
    tp = d.tau + delta; keep = d.ICULOS < tp; s = d[keep]; y = ((tp[keep] - s.ICULOS) <= 6).astype(int).to_numpy(); p = s.z2.to_numpy()
    ci = boot_auc(y, p, s.patient_id.to_numpy())
    out.append(dict(delta=delta, rows=int(len(s)), positives=int(y.sum()), prevalence=float(y.mean()), auroc=float(AUC(y, p)), lo=float(ci[0]), hi=float(ci[1]), auprc=float(AP(y, p)), lift=float(AP(y, p) / y.mean())))
    print(out[-1])
json.dump(out, open(f"{R}/results/label_timing_sensitivity.json", "w"), indent=1)
