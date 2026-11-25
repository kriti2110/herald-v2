import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open("scripts/11_stay_level.py").read().split("# ---- validation-only config selection")[0]; exec(src)
oof = np.zeros(n); cfg = GRID["s2"]
for k in range(5):
    te = np.flatnonzero(fold_of == k); tr = np.flatnonzero(fold_of != k); oof[te] = fit_pred(A1, tr, te, cfg)[0]
tz = np.zeros(n)
for a, b in (("A", "B"), ("B", "A")):
    tr = np.flatnonzero(hosp == a); te = np.flatnonzero(hosp == b); tz[te] = fit_pred(A1, tr, te, cfg)[0]
pd.DataFrame({PID: upid, "y": y, "z_cv": oof, "z_transfer": tz, "fold": fold_of, "hospital": hosp, "n_hours": A2["n_hours"].to_numpy()}).to_parquet("results/stay_level_oof.parquet")
print("saved; AUROC", AUC(y, oof), AUC(y, tz))
