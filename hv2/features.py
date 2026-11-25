"""Strictly causal feature pipeline for HERALD-v2.

Every feature at hour t is a function of rows 0..t of that patient ONLY.
Allowed operations: forward-fill (LOCF), trailing rolling windows
(window ends at t), expanding statistics, positive shifts, cumulative
sums. Forbidden: backward fill, centered/leading windows, per-stay
statistics over the full sequence, any global scaler fitted on all data.
No global scaling is used at all (tree models are scale-free); the
neural branch standardizes with training-fold statistics only.

`audit_prefix_invariance` (scripts/02_audit_prefix.py) verifies the claim
end to end: features computed on rows 0..t alone must equal row t of the
features computed on the whole stay.
"""
import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
CONT = VITALS + LABS
STATIC = ["Age", "Gender"]
CARE = ["ICULOS", "HospAdmTime", "Unit1", "Unit2"]  # care-pathway / workflow variables
PID, TIME, LABEL = "patient_id", "ICULOS", "SepsisLabel"
WINDOWS = (6, 24)


def _roll(loc: pd.Series, pid: pd.Series, w: int, how: str) -> np.ndarray:
    r = loc.groupby(pid, sort=False).rolling(w, min_periods=1)
    return getattr(r, how)().to_numpy()


def build_features(raw: pd.DataFrame, leaky_control: bool = False, extra: bool = False, baseline: bool = False):
    """Return (meta, X, groups).

    meta   : DataFrame [patient_id, ICULOS, SepsisLabel] in row order
    X      : DataFrame float32, one column per feature, same row order
    groups : dict group_name -> list of columns (for ablations)
    leaky_control: add a deliberately NON-causal feature (centered rolling
        mean of HR) so the audit can demonstrate it detects leakage.
    """
    df = raw.sort_values([PID, TIME], kind="stable").reset_index(drop=True)
    pid = df[PID]
    gb = df.groupby(PID, sort=False)
    pos = gb.cumcount().to_numpy().astype(np.float32)

    cols, groups = {}, {"value": [], "timing": [], "roll": [], "composite": [], "static": [], "care": []}

    def add(name, arr, grp):
        cols[name] = np.asarray(arr, dtype=np.float32)
        groups[grp].append(name)

    loc_all = {}
    for c in CONT:
        obs = df[c].notna()
        loc = gb[c].ffill()  # last observation carried forward: causal
        loc_all[c] = loc
        add(f"{c}", loc.to_numpy(), "value")

        last_pos = pd.Series(np.where(obs, pos, np.nan), index=df.index).groupby(pid, sort=False).ffill()
        add(f"{c}_stale", (pos - last_pos.to_numpy()), "timing")
        cnt = obs.astype(np.float32).groupby(pid, sort=False).cumsum().to_numpy()
        add(f"{c}_cnt", cnt, "timing")
        add(f"{c}_rate", cnt / (pos + 1.0), "timing")

    # trailing-window statistics + deltas on the LOCF series (vitals only)
    for c in VITALS:
        loc = loc_all[c]
        for w in WINDOWS:
            add(f"{c}_mean{w}", _roll(loc, pid, w, "mean"), "roll")
            add(f"{c}_std{w}", _roll(loc, pid, w, "std"), "roll")
            add(f"{c}_min{w}", _roll(loc, pid, w, "min"), "roll")
            add(f"{c}_max{w}", _roll(loc, pid, w, "max"), "roll")
            add(f"{c}_d{w}", (loc - loc.groupby(pid, sort=False).shift(w)).to_numpy(), "roll")
    # stay-so-far extremes (expanding, causal) for vitals and labs
    for c in CONT:
        loc = loc_all[c]
        lg = loc.groupby(pid, sort=False)
        add(f"{c}_xmin", lg.cummin().to_numpy(), "roll")
        add(f"{c}_xmax", lg.cummax().to_numpy(), "roll")
    # delta vs. 12 h ago for the labs that matter most clinically
    for c in ["Lactate", "Creatinine", "Platelets", "WBC", "Bilirubin_total", "BUN", "pH"]:
        loc = loc_all[c]
        add(f"{c}_d12", (loc - loc.groupby(pid, sort=False).shift(12)).to_numpy(), "roll")

    # composite physiology scores, all from LOCF values (causal)
    L = {c: loc_all[c].to_numpy() for c in CONT}
    with np.errstate(invalid="ignore", divide="ignore"):
        add("shock_index", L["HR"] / L["SBP"], "composite")
        add("pulse_pressure", L["SBP"] - L["DBP"], "composite")
        hr_hi = (L["HR"] > 90).astype(np.float32)
        rr_hi = (L["Resp"] > 20).astype(np.float32)
        t_ab = ((L["Temp"] > 38) | (L["Temp"] < 36)).astype(np.float32)
        wbc_ab = ((L["WBC"] > 12) | (L["WBC"] < 4)).astype(np.float32)
        add("sirs", hr_hi + rr_hi + t_ab + wbc_ab, "composite")
        add("qsofa", (L["Resp"] >= 22).astype(np.float32) + (L["SBP"] <= 100).astype(np.float32), "composite")
        add("map_low", (L["MAP"] < 65).astype(np.float32), "composite")
        add("spo2_low", (L["O2Sat"] < 92).astype(np.float32), "composite")
        dys = (
            (L["Lactate"] > 2).astype(np.float32) + (L["Creatinine"] > 2).astype(np.float32)
            + (L["Platelets"] < 100).astype(np.float32) + (L["Bilirubin_total"] > 2).astype(np.float32)
            + (L["MAP"] < 65).astype(np.float32) + (L["PaCO2"] > 50).astype(np.float32)
        )
        add("organ_dys_count", dys, "composite")
    sirs_s = pd.Series(cols["sirs"], index=df.index)
    add("sirs_hours_so_far", (sirs_s >= 2).astype(np.float32).groupby(pid, sort=False).cumsum().to_numpy(), "composite")
    add("sirs_mean6", _roll(sirs_s, pid, 6, "mean"), "composite")

    for c in STATIC:
        add(c, df[c].to_numpy(), "static")
    for c in CARE:
        add(c, df[c].to_numpy(), "care")

    if extra:  # ---- v2 additions: all trailing / expanding / positive-shift only ----
        for c in VITALS:
            loc = loc_all[c]
            add(f"{c}_mean3", _roll(loc, pid, 3, "mean"), "roll")
            add(f"{c}_mean12", _roll(loc, pid, 12, "mean"), "roll")
            m6 = pd.Series(cols[f"{c}_mean6"], index=df.index)
            add(f"{c}_slope6", ((m6 - m6.groupby(pid, sort=False).shift(6)) / 6).to_numpy(), "roll")
        for nm in ["sirs", "qsofa", "organ_dys_count"]:
            sr = pd.Series(cols[nm], index=df.index)
            add(f"{nm}_max6", _roll(sr, pid, 6, "max"), "composite")
            add(f"{nm}_max24", _roll(sr, pid, 24, "max"), "composite")
            add(f"{nm}_cummax", sr.groupby(pid, sort=False).cummax().to_numpy(), "composite")
        with np.errstate(invalid="ignore", divide="ignore"):
            flags = {"hr_hi": L["HR"] > 90, "rr_hi": L["Resp"] > 20, "temp_ab": (L["Temp"] > 38) | (L["Temp"] < 36),
                     "map_low": L["MAP"] < 65, "sbp_low": L["SBP"] < 100, "spo2_low": L["O2Sat"] < 92}
            for nm, fl in flags.items():
                sr = pd.Series(fl.astype(np.float32), index=df.index)
                add(f"{nm}_n6", _roll(sr, pid, 6, "sum"), "composite")
                add(f"{nm}_n24", _roll(sr, pid, 24, "sum"), "composite")
            first = {"map_low": L["MAP"] < 65, "lactate_hi": L["Lactate"] > 2, "sirs2": np.asarray(cols["sirs"]) >= 2}
            for nm, fl in first.items():
                fp = pd.Series(np.where(fl, pos, np.nan), index=df.index).groupby(pid, sort=False).cummin()
                fp = fp.groupby(pid, sort=False).ffill()
                add(f"t_since_first_{nm}", (pos - fp.to_numpy()), "composite")
            add("bun_creat", L["BUN"] / L["Creatinine"], "composite")
            add("sao2_fio2", L["SaO2"] / L["FiO2"], "composite")
        for c in ["Lactate", "Creatinine", "Platelets", "WBC", "Bilirubin_total", "BUN", "pH", "HCO3", "Glucose", "Hgb", "Potassium", "Temp"]:
            obs = df[c].notna(); loc = loc_all[c]
            prev = loc.groupby(pid, sort=False).shift(1).where(obs).groupby(pid, sort=False).ffill()
            add(f"{c}_dobs", (loc - prev).to_numpy(), "roll")
        nl = pd.Series(df[LABS].notna().sum(1).astype(np.float32), index=df.index)
        add("n_labs6", _roll(nl, pid, 6, "sum"), "timing")
        add("n_labs24", _roll(nl, pid, 24, "sum"), "timing")

    if baseline:  # ---- v3: deviation from the patient's OWN baseline (expanding / first-24h), causal ----
        BCH = VITALS + ["Lactate", "Creatinine", "Platelets", "WBC", "BUN", "Hgb", "pH", "HCO3", "Glucose", "Potassium", "Bilirubin_total", "Chloride"]
        for c in BCH:
            loc = loc_all[c]; ok = loc.notna().astype(np.float32)
            cnt = ok.groupby(pid, sort=False).cumsum()
            ssum = loc.fillna(0).groupby(pid, sort=False).cumsum(); sq = (loc.fillna(0) ** 2).groupby(pid, sort=False).cumsum()
            with np.errstate(invalid="ignore", divide="ignore"):
                em = ssum / cnt.where(cnt > 0)
                es = np.sqrt(np.maximum(sq / cnt.where(cnt > 0) - em ** 2, 0))
            b23 = em.where(pd.Series(pos == 23, index=df.index)).groupby(pid, sort=False).ffill()
            base = em.where(pd.Series(pos < 23, index=df.index), b23)
            add(f"{c}_dbase", (loc - base).to_numpy(), "roll")
            add(f"{c}_zexp", ((loc - em) / (es + 1e-3)).to_numpy(), "roll")

    if leaky_control:  # NEGATIVE CONTROL: uses future rows, must fail the audit
        lk = loc_all["HR"].groupby(pid, sort=False).transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
        add("LEAKY_HR_centered5", lk.to_numpy(), "roll")

    X = pd.DataFrame(cols, index=df.index)
    meta = df[[PID, TIME, LABEL]].copy()
    return meta, X, groups


def forecast_rows(meta: pd.DataFrame, h: int = 6):
    """Boolean keep-mask and 0/1 target for the h-hour first-positive-label
    forecast task (same definition as the original HERALD evaluation):
    septic stays contribute rows strictly before the first positive label,
    target = 1 iff that label falls within the next h hours; non-septic
    stays contribute every row with target 0."""
    pos = meta.groupby(PID, sort=False).cumcount().to_numpy()
    is_ev = meta[LABEL].to_numpy() > 0.5
    ev = np.where(is_ev, pos, np.iinfo(np.int64).max)
    onset = pd.Series(ev, index=meta.index).groupby(meta[PID], sort=False).transform("min").to_numpy()
    keep = pos < onset
    target = ((onset - pos) <= h) & keep
    return keep, target.astype(np.int8)
