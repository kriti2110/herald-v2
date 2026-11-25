"""Cohort characteristics by hospital (with standardised mean differences), participant-flow numbers, and the number of
patients and events used in each training / early-stopping / evaluation stage of the final two-stage pipeline.
Reads cached data and saved fold assignments only; nothing is trained. Splits replicate scripts/07_final.py exactly."""
import json, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
R = os.environ.get("HERALD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
meta = pd.read_parquet(f"{R}/cache/meta.parquet"); PID = "patient_id"
pc, upid = pd.factorize(meta[PID]); folds = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]
fold_of, ever, hosp = folds.fold.to_numpy(), folds.ever_septic.to_numpy(), folds.hospital.to_numpy()
out = {}
# ---- cohort characteristics ----
KEY = ["HR", "MAP", "Temp", "Resp", "WBC", "Lactate", "Creatinine", "Platelets"]
raw = pd.read_parquet(f"{R}/cache/raw_long.parquet", columns=[PID, "Age", "Gender", "ICULOS", "SepsisLabel"] + KEY)
raw["hosp"] = raw[PID].map(folds.reset_index().set_index(PID).hospital) if False else raw[PID].map(pd.Series(hosp, index=upid))
g = raw.groupby(PID, sort=False)
stay = pd.DataFrame({"hosp": g.hosp.first(), "Age": g.Age.first(), "Female": (g.Gender.first() == 0).astype(float), "los": g.size().astype(float), "septic": g.SepsisLabel.max().astype(float)})
for c in KEY: stay[c] = g[c].median()
def smd(a, b):
    sd = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2); return float(abs(np.nanmean(a) - np.nanmean(b)) / sd) if sd > 0 else 0.0
def q(x): x = x.dropna(); return [float(x.median()), float(x.quantile(.25)), float(x.quantile(.75))]
A, B = stay[stay.hosp == "A"], stay[stay.hosp == "B"]; rows = []
def add(name, col, kind):
    if kind == "bin": rows.append(dict(name=name, kind=kind, A=float(100 * A[col].mean()), B=float(100 * B[col].mean()), all=float(100 * stay[col].mean()), smd=smd(A[col], B[col])))
    else: rows.append(dict(name=name, kind=kind, A=q(A[col]), B=q(B[col]), all=q(stay[col]), smd=smd(A[col], B[col]), n_missing_stays=[int(A[col].isna().sum()), int(B[col].isna().sum())]))
add("Age (years)", "Age", "cont"); add("Female sex (%)", "Female", "bin"); add("ICU stay length (hours)", "los", "cont"); add("Septic stays (%)", "septic", "bin")
for c in KEY: add(f"{c}, per-stay median", c, "cont")
out["characteristics"] = rows
miss = []
for c in KEY:
    mA, mB, mAll = (raw[raw.hosp == "A"][c].isna().mean(), raw[raw.hosp == "B"][c].isna().mean(), raw[c].isna().mean())
    p = (mA + mB) / 2; s = np.sqrt((mA * (1 - mA) + mB * (1 - mB)) / 2)
    miss.append(dict(name=c, A=float(100 * mA), B=float(100 * mB), all=float(100 * mAll), smd=float(abs(mA - mB) / s)))
out["missing_rows_pct"] = miss
# ---- participant flow ----
fc = meta.fc_keep.to_numpy()
out["flow"] = dict(files_downloaded=40338, duplicate_files_removed=2, stays=int(len(upid)), stays_A=int((hosp == "A").sum()), stays_B=int((hosp == "B").sum()),
    septic_stays=int(ever.sum()), non_septic_stays=int((~ever.astype(bool)).sum()), rows=int(len(meta)), forecast_rows=int(fc.sum()),
    forecast_positive_rows=int(meta.fc_target.to_numpy()[fc].sum()), detect_positive_rows=int(meta.det_target.sum()),
    patients_per_fold=[int((fold_of == k).sum()) for k in range(5)])
# ---- per-stage counts (replicates 07_final.py splits) ----
Tf, Td = meta.fc_target.to_numpy(), meta.det_target.to_numpy()
def rows_mask(pats, task):
    m = np.zeros(len(upid), bool); m[pats] = True; m = m[pc]
    if task == "fc": m &= fc
    return m
def count(pats):
    return dict(patients=len(pats), septic=int(ever[pats].sum()), fc_events=int(Tf[rows_mask(pats, "fc")].sum()), det_events=int(Td[rows_mask(pats, "det")].sum()))
acc = {k: [] for k in ["outer_train", "stage1_fit", "stage1_early_stop", "stage1_oof_scored", "stage2_fit", "stage2_early_stop", "test_fold"]}
allp = np.arange(len(upid))
for k in range(5):
    P, Te = allp[fold_of != k], allp[fold_of == k]
    acc["outer_train"].append(count(P)); acc["test_fold"].append(count(Te))
    for j, (a, b) in enumerate(StratifiedKFold(4, shuffle=True, random_state=1).split(P, ever[P])):
        a_tr, a_va = train_test_split(P[a], test_size=0.12, stratify=ever[P[a]], random_state=j)
        acc["stage1_fit"].append(count(a_tr)); acc["stage1_early_stop"].append(count(a_va)); acc["stage1_oof_scored"].append(count(P[b]))
    p2, v2 = train_test_split(P, test_size=0.15, stratify=ever[P], random_state=7)
    acc["stage2_fit"].append(count(p2)); acc["stage2_early_stop"].append(count(v2))
out["stage_counts_mean"] = {s: {m: float(np.mean([x[m] for x in v])) for m in v[0]} | {"n_models": len(v)} for s, v in acc.items()}
json.dump(out, open(f"{R}/results/cohort_and_counts.json", "w"), indent=1)
print(json.dumps(out["flow"])); print(json.dumps(out["stage_counts_mean"], indent=0)[:1400])
for r in rows: print(r["name"], r["A"], r["B"], round(r["smd"], 3))
