"""End-to-end prefix-invariance audit of the raw -> features pipeline.

For sampled stays and several cut hours t, features computed from rows 0..t
ALONE must equal row t of the features computed on the WHOLE stay. A
deliberately non-causal feature (centered rolling mean) is included as a
negative control: the audit must flag it, otherwise it has no power.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from hv2.features import build_features, PID, TIME

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = pd.read_parquet(f"{R}/cache/raw_long.parquet")
meta = pd.read_parquet(f"{R}/cache/meta.parquet")
rng = np.random.default_rng(0)
ever = meta.groupby(PID)["SepsisLabel"].max()
pids = list(rng.choice(ever[ever > 0].index, 100, replace=False)) + list(rng.choice(ever[ever == 0].index, 100, replace=False))
rawg = {p: g for p, g in raw.groupby(PID)}
RTOL, ATOL = 1e-4, 1e-4
bad, n_cmp, n_cut = {}, 0, 0
t0 = time.time()
for i, p in enumerate(pids):
    g = rawg[p].sort_values(TIME).reset_index(drop=True)
    _, Xf, _ = build_features(g, leaky_control=True, extra=True, baseline=True)
    L = len(g)
    cuts = sorted({0, 1, 2, 5, 11, 23, L // 2, L - 2, L - 1} & set(range(L)))
    for t in cuts:
        _, Xp, _ = build_features(g.iloc[: t + 1], leaky_control=True, extra=True, baseline=True)
        a, b = Xp.iloc[-1].to_numpy(np.float64), Xf.iloc[t].to_numpy(np.float64)
        ok = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=RTOL, atol=ATOL)
        for c in np.array(Xf.columns)[~ok]:
            bad[c] = bad.get(c, 0) + 1
        n_cmp += len(a); n_cut += 1
    if i % 25 == 0:
        print(f"{i}/{len(pids)} patients, {n_cut} cuts, {time.time()-t0:.0f}s", flush=True)
leak = {k: v for k, v in bad.items() if k.startswith("LEAKY")}
real = {k: v for k, v in bad.items() if not k.startswith("LEAKY")}
out = {
    "n_patients": len(pids), "n_prefix_cuts": n_cut, "n_feature_comparisons": n_cmp,
    "n_causal_features": int(Xf.shape[1]) - 1, "tolerance": {"rtol": RTOL, "atol": ATOL},
    "causal_features_violating": real,
    "PASS_all_causal_features": len(real) == 0,
    "negative_control_LEAKY_HR_centered5_violations": leak.get("LEAKY_HR_centered5", 0),
    "negative_control_detected": leak.get("LEAKY_HR_centered5", 0) > 0,
}
json.dump(out, open(f"{R}/results/prefix_audit_v3.json", "w"), indent=2)
print(json.dumps(out, indent=2))
