"""Fig. 21 (decision curves) and tables for subgroups and alert burden, from results/subgroups_dca_burden.json."""
import json, numpy as np
from figstyle import *
J = json.load(open(f"{R}/results/subgroups_dca_burden.json"))
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=False)
for ax, (task, col), pl in zip(axs, [("Early 6 h", TASK_COL["forecast"]), ("Detection", TASK_COL["detect"])], "ab"):
    d = J["dca"][task]; th = np.array(d["thresholds"]); nb = np.array(d["net_benefit"]); ta = np.array(d["treat_all"])
    ax.plot(th, nb * 1e3, color=col, lw=1.8, label="HERALD (calibrated)")
    ax.plot(th, np.maximum(ta, -1) * 1e3, color=C["slate"], lw=1.2, ls="--", label="Treat all")
    ax.axhline(0, color=C["ink"], lw=0.9, label="Treat none")
    ax.set_xlim(0, 0.15); ax.set_ylim(-2, max(nb.max(), 0) * 1e3 * 1.15)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit (per 1000 rows)"); ax.set_title(task); panel(ax, pl)
    ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig21_decision_curve.png"); print("fig ok")
LV = {'<45': '$<$45', '45-64': '45--64', '65-79': '65--79', '>=80': '$\\geq$80'}
# subgroup table
rows = []; last = None
for task in ["Early 6 h", "Detection"]:
    rows.append("\\midrule" if rows else "")
    for s in [x for x in J["subgroups"] if x["task"] == task]:
        g = s["group"] if s["group"] != last else ""; last = s["group"]
        rows.append(f"{task if g and s['group']=='Age (years)' else ''} & {g} & {LV.get(s['level'], s['level'])} & {s['n_patients']:,} & {100*s['prevalence']:.2f} & {s['auroc']:.3f} [{s['auroc_lo']:.3f}, {s['auroc_hi']:.3f}] & {s['auprc']:.3f} & {s['lift']:.1f} \\\\".replace(f"{s['n_patients']:,}", f"{s['n_patients']:,}".replace(",", "{,}")))
    last = None
t = ("\\begin{table}[!t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Subgroup performance of the final model (pooled out-of-fold, natural prevalence). AUROC with 95\\% patient-clustered bootstrap intervals (300 replicates); AUPRC is a point estimate and lift is AUPRC over the subgroup's own prevalence. Age bands in years; sex as coded in the challenge data.}\n"
     "\\label{tab:subgroups}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{llcrcccc}\n\\toprule\nTask & Group & Level & Patients & Prev.\\ (\\%) & AUROC [95\\% CI] & AUPRC & Lift \\\\\n\\midrule\n"
     + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
open(f"{TAB}/t_subgroups.tex", "w").write(t)
# alert burden table
b = J["burden"]; L = []
for task in ["Early 6 h", "Detection"]:
    for sp in (90, 95):
        v = b[f"{task}|spec{sp}"]
        L.append(f"{task} & {sp}\\% & {v['false_alarms_per_patient_day']:.2f} & {v['alerts_per_tp']:.1f} & {100*v['septic_alerted_before_label']/v['septic_stays']:.0f}\\% & {v['lead_median']:.0f} [{v['lead_q1']:.0f}--{v['lead_q3']:.0f}] \\\\")
t = ("\\begin{table}[!t]\n\\centering\\scriptsize\n\\caption{Alert burden at fixed row-level specificity (pooled out-of-fold; thresholds set on the same predictions, so illustrative). False alarms are alerts on rows with a negative target per 24 patient-hours. Lead time is the time from the first alert to the first positive label, median [IQR] over septic stays alerted before the label.}\n"
     "\\label{tab:burden}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{lccccc}\n\\toprule\nTask & Spec. & False alarms / patient-day & Alerts / TP & Septic stays alerted & Lead time (h) \\\\\n\\midrule\n"
     + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
open(f"{TAB}/t_burden.tex", "w").write(t)
for task in ["Early 6 h", "Detection"]:
    d = J["dca"][task]; th = np.array(d["thresholds"]); nb = np.array(d["net_benefit"]); ta = np.array(d["treat_all"])
    ok = th[(nb > np.maximum(ta, 0))]; print(task, "NB>max(all,none) at", ok.min() if len(ok) else None, "to", ok.max() if len(ok) else None, "prev", d["prevalence"], "nb@1%", nb[th==0.01], "ta@1%", ta[th==0.01])
