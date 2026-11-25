"""Build causal features for all stays once; cache to disk."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from hv2.features import build_features, forecast_rows, PID

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
raw = pd.read_parquet(f"{R}/cache/raw_long.parquet")
meta, X, groups = build_features(raw, extra=True, baseline=True)
print(f"features {X.shape} in {time.time()-t0:.0f}s", flush=True)
keep, tgt = forecast_rows(meta, 6)
meta["hospital"] = np.where(meta[PID].str.startswith("p0"), "A", "B")
meta["fc_keep"] = keep
meta["fc_target"] = tgt
meta["det_target"] = (meta["SepsisLabel"] > 0.5).astype(np.int8)
m0=pd.read_parquet(f"{R}/cache/meta.parquet"); assert (m0[PID].to_numpy()==meta[PID].to_numpy()).all() and (m0["ICULOS"].to_numpy()==meta["ICULOS"].to_numpy()).all(), "row order changed"
np.save(f"{R}/cache/X3.npy", X.to_numpy(dtype=np.float32))
json.dump({"columns": list(X.columns), "groups": groups}, open(f"{R}/cache/columns3.json", "w"))
print("n_features", X.shape[1], {k: len(v) for k, v in groups.items()})
print("forecast rows", int(keep.sum()), "pos", int(tgt.sum()), f"prev={tgt[keep].mean():.4%}")
print("detection rows", len(meta), "pos", int(meta.det_target.sum()), f"prev={meta.det_target.mean():.4%}")
print("patients", meta[PID].nunique(), "A", meta[meta.hospital=='A'][PID].nunique(), "B", meta[meta.hospital=='B'][PID].nunique())
