"""Stage-2 history features and the two-stage inference path (shared by training of the release models and by prediction)."""
import json, os
import numpy as np, pandas as pd


def hist_feats(codes, z, thr):
    """Causal history of a score series. `codes` = contiguous patient codes (rows in time order). Identical to scripts/07_final.py."""
    s = pd.Series(z); gb = s.groupby(codes); f = {"z": z, "z_cummax": gb.cummax().to_numpy()}
    for w in (3, 6, 12, 24): f[f"z_mean{w}"] = gb.rolling(w, min_periods=1).mean().to_numpy()
    for w in (6, 24): f[f"z_max{w}"] = gb.rolling(w, min_periods=1).max().to_numpy()
    f["dz3"] = (s - gb.shift(3)).to_numpy(); f["dz6"] = (s - gb.shift(6)).to_numpy()
    for nm, t in thr.items():
        fl = pd.Series((z > t).astype(np.float32)).groupby(codes)
        f[f"n_{nm}_6"] = fl.rolling(6, min_periods=1).sum().to_numpy(); f[f"n_{nm}_24"] = fl.rolling(24, min_periods=1).sum().to_numpy()
    return pd.DataFrame(f)


def predict(model_dir, X, patient_codes, iculos):
    """Return the final two-stage risk logit z2 for each row.
    X: DataFrame/array with the 370 feature columns in the order given by model_dir/model_info.json ('columns');
    patient_codes: contiguous integer code per patient (rows in time order); iculos: ICULOS per row."""
    import lightgbm as lgb
    info = json.load(open(os.path.join(model_dir, "model_info.json"))); cols = info["columns"]
    Xa = np.asarray(X[cols] if hasattr(X, "columns") else X, dtype=np.float32)
    z = np.mean([lgb.Booster(model_file=os.path.join(model_dir, f"stage1_inner{j}.txt")).predict(Xa, raw_score=True) for j in range(4)], axis=0)
    F = hist_feats(patient_codes, z, info["thresholds"]); F["ICULOS"] = iculos
    for c in info["top25"]: F[c] = Xa[:, cols.index(c)]
    return lgb.Booster(model_file=os.path.join(model_dir, "stage2.txt")).predict(F, raw_score=True)
