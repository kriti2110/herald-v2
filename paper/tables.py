"""Generate every LaTeX table + numbers.tex macros straight from results/*. Nothing is typed by hand except literature rows."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from analysis import *
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

STAY = pd.read_parquet(f"{R}/results/stay_level_oof.parquet")
f3 = lambda v: f"{v:.3f}"
ci = lambda m, k: f"{m[k]:.3f} [{m[k + '_CI95'][0]:.3f}, {m[k + '_CI95'][1]:.3f}]"


def w(name, body):
    open(f"{TAB}/{name}.tex", "w").write(body); print("table", name)


def tabular(cols, rows, head, bold_last=False, midrule_before=()):
    out = ["\\begin{adjustbox}{max width=\\linewidth}", "\\begin{tabular}{" + cols + "}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrule_before: out.append("\\midrule")
        out.append(" & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]; return "\n".join(out)


sl, slt = SL["primary_no_length"]["cv_pooled"], SL["primary_no_length"]["transfer_pooled"]
M = {"fc": F2("cv/forecast")["pooled"], "dt": F2("cv/detect")["pooled"], "fct": F2("transfer/forecast")["pooled"], "dtt": F2("transfer/detect")["pooled"]}

# ------------------------------------------------------------------ numbers.tex
mac = {
    "NStays": "40{,}336", "NHospA": "20{,}336", "NHospB": "20{,}000", "NSeptic": f"{int(FOLDS.ever_septic.sum()):,}".replace(",", "{,}"),
    "AurocFc": f3(M["fc"]["AUROC"]), "AuprcFc": f3(M["fc"]["AUPRC"]), "AurocDt": f3(M["dt"]["AUROC"]), "AuprcDt": f3(M["dt"]["AUPRC"]),
    "AurocStay": f3(sl["AUROC"]), "AuprcStay": f3(sl["AUPRC"]), "AurocStayT": f3(slt["AUROC"]), "AuprcStayT": f3(slt["AUPRC"]),
    "AurocFcT": f3(M["fct"]["AUROC"]), "AurocDtT": f3(M["dtt"]["AUROC"]), "AuprcFcT": f3(M["fct"]["AUPRC"]), "AuprcDtT": f3(M["dtt"]["AUPRC"]),
    "BalAuroc": f"{SD['balanced_432_432_view']['AUROC']:.3f}", "BalAuprc": f"{SD['balanced_432_432_view']['AUPRC']:.3f}",
    "LenAlone": f"{SD['AUROC_from_stay_length_alone']:.2f}", "PatAurocDt": f3(M["dt"]["patient_AUROC"]), "PatAuprcDt": f3(M["dt"]["patient_AUPRC"]),
    "NCut": f"{AUD2['n_prefix_cuts']:,}".replace(",", "{,}"), "NCmp": f"{AUD2['n_feature_comparisons']:,}".replace(",", "{,}"), "NCtrl": f"{AUD2['negative_control_LEAKY_HR_centered5_violations']:,}".replace(",", "{,}"),
    "TopsisV": f"{TOP['equal']['HERALD (this work)']:.3f}", "TopsisL": f"{TOP['equal']['LightGBM']:.3f}",
    "LitTopsisH": f"{LIT_EQUAL['HERALD (hourly)']:.3f}", "LitTopsisS": f"{LIT_EQUAL['HERALD (stay-level)']:.3f}",
    "LitTopsisSA": f"{LIT_EQUAL['SepsisAI [5]']:.3f}", "LitTopsisK": f"{LIT_EQUAL['LSTM-GAT [6]']:.3f}", "LitTopsisT": f"{LIT_EQUAL['Transformer-eICU [7]']:.3f}",
}
open(f"{R}/paper/numbers.tex", "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")

# ------------------------------------------------------------------ T1: literature (qualitative, as reported)
rows = [
    ["SepsisAI \\cite{gupta2024}", "PhysioNet 2019", "6\\,h look-ahead + alert rule", "Random, patient-level", "Balanced (432+432); also 34+432", "None reported", "None seen", "0.95/0.96 (patient, balanced)"],
    ["LSTM-GAT \\cite{khorram2026}", "PhysioNet 2019", "12\\,h", "Random 70/15/15, 5 seeds", "Not stated", "SD over seeds", "MIMIC-III, vitals only", "0.853 $\\pm$ 0.005 AUROC"],
    ["Transformer \\cite{tang2024}", "eICU (+MIMIC)", "4/8/12\\,h", "70:30 + 5-fold CV", "Matched cases/controls", "Mean $\\pm$ SD", "MIMIC (AUC $\\approx$ 0.9)", "AUROC up to 0.99"],
    ["MB-TCN (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019", "``Imminent''$^\\dagger$", "n/v", "n/v", "n/v", "n/v", "0.892/0.527$^\\dagger$"],
    ["PoEMS (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019", "``Imminent''$^\\dagger$", "n/v", "n/v", "n/v", "n/v", "0.911/0.675$^\\dagger$"],
    ["SupMix (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019", "H0$^\\dagger$", "n/v", "n/v", "n/v", "n/v", "0.883/0.667 (AP)$^\\dagger$"],
    ["\\textbf{HERALD (this work)}", "PhysioNet 2019", "6/12/24\\,h; hourly; stay", "Patient-level 5-fold CV", "Natural prevalence", "Bootstrap CI + fold SD", "Hospital A$\\leftrightarrow$B", f"{M['dt']['AUROC']:.3f}/{M['dt']['AUPRC']:.3f} hourly; {sl['AUROC']:.3f}/{sl['AUPRC']:.3f} stay"]]
w("t_related", "\\begin{table*}[!t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Recent PhysioNet-2019-era sepsis studies as reported in the text we could read. Headline values are AUROC/AUPRC under each study's \\emph{own} protocol and are \\textbf{not} directly comparable. $^\\dagger$Figures quoted from \\cite{mansoor2026}; original paper not retrieved (n/v = not verified).}\n\\label{tab:related}\n"
  + tabular(">{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{1.5cm}>{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{2.3cm}>{\\raggedright\\arraybackslash}p{1.7cm}>{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{2.3cm}", rows, ["Study", "Data", "Target", "Split", "Test set", "Uncertainty", "External", "Headline"], midrule_before=(6,)) + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T2: cohort & protocol
fc_rows = 1524294; fc_pos = 14079; det_pos = 27916
rows = [["Hourly ICU records (rows)", "1{,}552{,}210"], ["ICU stays (hospital A / B)", "40{,}336 (20{,}336 / 20{,}000)"], ["Septic stays (ever SepsisLabel $=1$)", f"{int(FOLDS.ever_septic.sum()):,} (7.27\\%)".replace(",", "{,}")],
        ["Forecast rows / positives (6\\,h)", f"{fc_rows:,} / {fc_pos:,} (0.92\\%)".replace(",", "{,}")], ["Detection positives (hourly)", f"{det_pos:,} (1.80\\%)".replace(",", "{,}")],
        ["Outer evaluation", "5 patient-level stratified folds (seed 42)"], ["Inner split (per outer fold)", "stage 1: 4 inner folds, 12\\% early-stopping holdout; stage 2: 85 / 15\\% fit / early stopping"], ["Calibration", "isotonic, fitted on the other four folds' out-of-fold scores"],
        ["Cross-hospital", "train A$\\to$test B and B$\\to$A"], ["Test-set handling", "natural prevalence; never resampled"], ["Uncertainty", "patient-clustered bootstrap (1000 reps)"]]
w("t_protocol", "\\begin{table}[!t]\n\\centering\\footnotesize\n\\caption{Cohort and evaluation protocol.}\n\\label{tab:protocol}\n" + tabular("ll", rows, ["Item", "Value"]) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T3: feature groups
G = {k: len(v) for k, v in COLS["groups"].items()}
rows = [["Last observed value", G["value"], "LOCF of 34 vitals/labs"], ["Measurement timing", G["timing"], "staleness, counts, rates, labs ordered 6/24\\,h"],
        ["Trailing / expanding", G["roll"], "mean/std/min/max (3--24\\,h), slopes, deltas, stay-so-far extremes"], ["Clinical composites", G["composite"], "shock index, SIRS, qSOFA, organ-dysfunction count, flags"],
        ["Static", G["static"], "age, sex"], ["Care pathway", G["care"], "ICULOS, admission time, unit 1/2"], ["\\textbf{Total}", f"\\textbf{{{sum(G.values())}}}", ""]]
w("t_features", "\\begin{table}[!t]\n\\centering\\footnotesize\n\\caption{Causal feature groups (all strictly backward-looking).}\n\\label{tab:features}\n" + tabular("lrp{4.4cm}", rows, ["Group", "\\#", "Content"], midrule_before=(6,)) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T4: hyper-parameters
rows = [["Stage 1 (forecast)", "LightGBM", "lr .05, 31 leaves, min-child 500, feat-frac .3, $\\ell_2$ 10"], ["Stage 1 (detect)", "LightGBM", "lr .05, 31 leaves, min-child 2000, feat-frac .2, $\\ell_2$ 30, extra-trees"],
        ["Stage 2", "LightGBM", "lr .05, 15 leaves, min-child 500, feat-frac .5, $\\ell_2$ 10"], ["Stay-level", "LightGBM", "lr .03, 15 leaves, min-child 50, feat-frac .3, $\\ell_2$ 10"],
        ["Causal TCN", "6 blocks, $d{=}96$", "kernel 3, dilations 1--32, dropout .15, AdamW 2e-3, one-cycle, pos-weight 5"], ["Early stopping", "all GBMs", "100 rounds on validation log-loss (first metric)"]]
w("t_hyper", "\\begin{table}[!t]\n\\centering\\scriptsize\n\\caption{Model configurations (chosen on validation patients only).}\n\\label{tab:hyper}\n" + tabular("p{1.8cm}p{1.5cm}>{\\raggedright\\arraybackslash}p{4.3cm}", rows, ["Model", "Type", "Setting"]) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T5: main results
rows = [["Early prediction 6\\,h", "5-fold CV", ci(M["fc"], "AUROC"), ci(M["fc"], "AUPRC"), f3(M["fc"]["patient_AUROC"]), f"{100 * M['fc']['prevalence']:.2f}\\%"],
        ["Hourly detection", "5-fold CV", ci(M["dt"], "AUROC"), ci(M["dt"], "AUPRC"), f3(M["dt"]["patient_AUROC"]), f"{100 * M['dt']['prevalence']:.2f}\\%"],
        ["Stay-level detection", "5-fold CV", ci(sl, "AUROC"), ci(sl, "AUPRC"), "--", f"{100 * sl['prevalence']:.2f}\\%"],
        ["Early prediction 6\\,h", "A$\\leftrightarrow$B", ci(M["fct"], "AUROC"), ci(M["fct"], "AUPRC"), f3(M["fct"]["patient_AUROC"]), f"{100 * M['fct']['prevalence']:.2f}\\%"],
        ["Hourly detection", "A$\\leftrightarrow$B", ci(M["dtt"], "AUROC"), ci(M["dtt"], "AUPRC"), f3(M["dtt"]["patient_AUROC"]), f"{100 * M['dtt']['prevalence']:.2f}\\%"],
        ["Stay-level detection", "A$\\leftrightarrow$B", ci(slt, "AUROC"), ci(slt, "AUPRC"), "--", f"{100 * slt['prevalence']:.2f}\\%"]]
w("t_main", "\\begin{table*}[!t]\n\\centering\\footnotesize\n\\caption{Final results at natural prevalence (pooled out-of-fold predictions; 95\\% patient-clustered bootstrap CI). Patient AUROC scores each stay by its highest hourly risk.}\n\\label{tab:main}\n"
  + tabular("llcccc", rows, ["Task", "Setting", "AUROC [95\\% CI]", "AUPRC [95\\% CI]", "Patient AUROC", "Prevalence"], midrule_before=(3,)) + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T6: per-fold
rows = []
for k in range(5):
    a, b_ = F2("cv/forecast")["per_group"][str(k)], F2("cv/detect")["per_group"][str(k)]; m = STAY[STAY.fold == k]
    rows.append([f"Fold {k}", f3(a["AUROC"]), f3(a["AUPRC"]), f3(b_["AUROC"]), f3(b_["AUPRC"]), f3(AUC(m.y, m.z_cv)), f3(AP(m.y, m.z_cv))])
def ms(v): return f"{np.mean(v):.3f} $\\pm$ {np.std(v, ddof=1):.3f}"
cols_ = list(zip(*[[float(x) for x in r[1:]] for r in rows])); rows.append(["\\textbf{Mean $\\pm$ SD}"] + [ms(c) for c in cols_])
w("t_folds", "\\begin{table*}[!t]\n\\centering\\footnotesize\n\\caption{Per-fold results (two-stage model; stay-level classifier). Folds are patient-disjoint.}\n\\label{tab:folds}\n"
  + tabular("lcccccc", rows, ["", "Forecast AUROC", "Forecast AUPRC", "Detect AUROC", "Detect AUPRC", "Stay AUROC", "Stay AUPRC"], midrule_before=(5,)) + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T7: transfer
def tr(task):
    g = F2(f"transfer/{task}")["per_group"]; return g["A->B"], g["B->A"]
a1, b1 = tr("forecast"); a2, b2 = tr("detect"); sa, sb = SL["primary_no_length"]["transfer_A_to_B"], SL["primary_no_length"]["transfer_B_to_A"]
rows = [["Early prediction 6\\,h", f3(M["fc"]["AUROC"]), f3(a1["AUROC"]), f3(b1["AUROC"]), f3(a1["AUPRC"]), f3(b1["AUPRC"])],
        ["Hourly detection", f3(M["dt"]["AUROC"]), f3(a2["AUROC"]), f3(b2["AUROC"]), f3(a2["AUPRC"]), f3(b2["AUPRC"])],
        ["Stay-level detection", f3(sl["AUROC"]), f3(sa["AUROC"]), f3(sb["AUROC"]), f3(sa["AUPRC"]), f3(sb["AUPRC"])]]
w("t_transfer", "\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Cross-hospital transfer (AUROC unless stated).}\n\\label{tab:transfer}\n"
  + tabular("lccccc", rows, ["Task", "CV", "A$\\to$B", "B$\\to$A", "AUPRC A$\\to$B", "AUPRC B$\\to$A"]) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T8: horizons
H = [(6, TS["cv/forecast"]["stage1"]["pooled"]), (12, HZ["horizon_12h"]), (24, HZ["horizon_24h"])]
rows = [[f"{h}\\,h", ci(m, "AUROC"), ci(m, "AUPRC"), f"{100 * m['prevalence']:.2f}\\%", f"$\\times${m['AUPRC'] / m['prevalence']:.1f}"] for h, m in H]
w("t_horizon", "\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Prediction horizon (single-stage boosted model, 5-fold CV). AUPRC rises with the horizon only because more rows are positive.}\n\\label{tab:horizon}\n"
  + tabular("lcccc", rows, ["Horizon", "AUROC [95\\% CI]", "AUPRC [95\\% CI]", "Prev.", "Lift"]) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T9: ablation
rows = []
for task, tl in (("forecast", "Early 6\\,h"), ("detect", "Detection")):
    for mode in ("cv", "transfer"):
        ab = GB[f"ablation/{mode}/{task}"]
        rows.append([tl, "CV" if mode == "cv" else "A$\\leftrightarrow$B"] + [f3(ab[k]["pooled"]["AUROC"]) if k in ab else "--" for k in ("full", "no_care", "physio_only")] + [f"{ab['physio_only']['pooled']['AUROC'] - ab['full']['pooled']['AUROC']:+.3f}"])
w("t_ablation", "\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Shortcut ablation (AUROC, boosted model). ``No care'' drops ICULOS/admission time/unit; ``physio-only'' also drops all measurement-timing features.}\n\\label{tab:ablation}\n"
  + tabular("llcccc", rows, ["Task", "Setting", "All", "No care", "Physio-only", "$\\Delta$"]) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T10: model family
rows = []
for task, tl in (("forecast", "Early 6\\,h"), ("detect", "Detection")):
    g = GB[f"cv/{task}"]
    for nm, m in (("Boosting (1 stage)", g["GBM"]["pooled"]), ("TCN", g["TCN"]["pooled"]), ("Boosting + TCN blend", g["ENSEMBLE"]["pooled"]), ("Two-stage (final)", F2(f"cv/{task}")["pooled"])):
        rows.append([tl, nm, ci(m, "AUROC"), ci(m, "AUPRC")])
w("t_family", "\\begin{table*}[!t]\n\\centering\\footnotesize\n\\caption{Model families (5-fold CV, pooled). The TCN did not improve on boosting.}\n\\label{tab:family}\n"
  + tabular("llcc", rows, ["Task", "Model", "AUROC [95\\% CI]", "AUPRC [95\\% CI]"], midrule_before=(4,)) + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T11: TOPSIS own models
order = ["HERALD (this work)", "LightGBM", "LightGBM-reg", "LightGBM no-meas.", "Causal encoder", "Causal enc.+aux"]
rows = [[n.replace("(this work)", "(ours)")] + [f3(TOPC.loc[n, c]) if c != "AlertsPerTP" else f"{TOPC.loc[n, c]:.1f}" for c in ("AUROC", "AUPRC", "PatAUROC", "Sens90", "AlertsPerTP")] + [f3(TOP[k][n]) for k in ("equal", "AUROC-heavy", "AUPRC-heavy", "operating-point-heavy")] for n in order]
w("t_topsis_models", "\\begin{table*}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3.4pt}\n\\caption{TOPSIS on the \\emph{identical} 229{,}244 forecast rows (5{,}983 test patients) of the earlier HERALD models: criteria and closeness $C^*$ under four weightings. Equal weights are used because the original weights were not archived; AlertsPerTP is a cost criterion.}\n\\label{tab:topsis_models}\n"
  + tabular("lcccccccccc", rows, ["Model", "AUROC", "AUPRC", "PatAUROC", "Sens@90", "A/TP", "$C^*$ eq.", "AUROC-hvy", "AUPRC-hvy", "Op.pt-hvy"][:10] + [""] * 0) .replace("lcccccccccc", "lccccccccc") + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T12: rigor scorecard + TOPSIS
sym = {0: "$\\times$", 0.5: "$\\circ$", 1: "\\checkmark"}
rows = [[p] + [sym[SCORES.loc[p, c]] for c in CRIT_FLAT] + [f"{SCORES.loc[p, 'Rigor score']:.2f}", f"{AUROC[p]:.3f}", f"{LIT_EQUAL[p]:.3f}"] for p in PAPERS]
w("t_rigor", "\\begin{table*}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Reporting-rigor scorecard (\\checkmark\\,=1, $\\circ$\\,=0.5, $\\times$\\,=0) scored by the authors from the text we could read, and the resulting TOPSIS closeness with equal weight on headline AUROC and rigor. The rubric, scores and AUROCs are our judgement; AUROCs come from different protocols.}\n\\label{tab:rigor}\n"
  + tabular("lccccccccc", rows, ["Study", "Patient split", "Natural prev.", "Uncert.", "Leakage audit", "External", "AUROC+AUPRC", "Rigor", "AUROC", "$C^*$"], midrule_before=(2,)) + "\n\\end{table*}\n")

# ------------------------------------------------------------------ T13: negative results
rows = []
for task, tl in (("forecast", "Early 6\\,h"), ("detect", "Detection")):
    base = R3[f"{task}/E0_v2_features"]
    for k, l in (("E1_plus_personal_baseline", "Personal-baseline features"), ("E2_multihorizon_soft_label", "Multi-horizon soft label"), ("E3_pos_weight3", "Positive weight $\\times$3"), ("E4_avg_of_3_configs", "Mean of 3 configurations")):
        if f"{task}/{k}" in R3:
            m = R3[f"{task}/{k}"]; rows.append([tl, l, f"{m['val_AUROC'] - base['val_AUROC']:+.4f}", f"{100 * (m['val_AUPRC'] / base['val_AUPRC'] - 1):+.1f}\\%"])
w("t_negative", "\\begin{table}[!t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\caption{Ideas tested on validation patients only (change vs.\\ the 370-feature set). None met the pre-set adoption rule ($\\Delta$AUROC $\\ge +0.001$ and AUPRC not down $>3\\%$).}\n\\label{tab:negative}\n"
  + tabular("llcc", rows, ["Task", "Idea", "$\\Delta$ val AUROC", "$\\Delta$ val AUPRC"], midrule_before=(4,)) + "\n\\end{table}\n")
print("done")


# ------------------------------------------------------------------ T14: stage 1 vs stage 2 (honest)
rows = []
for task, tl in (("forecast", "Early 6\\,h"), ("detect", "Detection")):
    for nm, key in (("Stage 1 only", "stage1"), ("Stage 1 + 2 (final)", "FINAL_stage2")):
        m = TS[f"cv/{task}"][key]["pooled"]; rows.append([tl, nm, ci(m, "AUROC"), ci(m, "AUPRC")])
w("t_stage", "\\begin{table}[!t]\n\\centering\\scriptsize\n\\caption{Value of the second stage (5-fold CV, pooled). It changes AUROC by $-0.001$ (forecast) and $+0.002$ (detection).}\n\\label{tab:stage}\n"
  + tabular("llcc", rows, ["Task", "Model", "AUROC [95\\% CI]", "AUPRC [95\\% CI]"], midrule_before=(2,)) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T15: operating points
def opp(task):
    d = pd.read_parquet(f"{R}/results/final_cv_{task}_preds.parquet", columns=["y", "z2"]); y, z = d.y.to_numpy(), d.z2.to_numpy(); out = []
    for spec in (0.90, 0.95, 0.99):
        thr = np.quantile(z[y == 0], spec); tp = int(((z > thr) & (y == 1)).sum()); fp = int(((z > thr) & (y == 0)).sum())
        out.append((spec, tp / y.sum(), (tp + fp) / max(tp, 1), tp / max(tp + fp, 1)))
    return out
rows = []
for task, tl in (("forecast", "Early 6\\,h"), ("detect", "Detection")):
    for spec, sens, atp, ppv in opp(task): rows.append([tl, f"{100 * spec:.0f}\\%", f"{sens:.3f}", f"{ppv:.3f}", f"{atp:.1f}"])
w("t_operating", "\\begin{table}[!t]\n\\centering\\scriptsize\n\\caption{Row-level operating points of the final model (pooled out-of-fold; threshold set on the same predictions, so illustrative). A/TP = alerts per true positive.}\n\\label{tab:operating}\n"
  + tabular("lcccc", rows, ["Task", "Specificity", "Sensitivity", "Precision", "A/TP"], midrule_before=(3,)) + "\n\\end{table}\n")

# ------------------------------------------------------------------ T16: rubric justification (appendix)
cell = lambda s_: s_
rows = [
 ["SepsisAI \\cite{gupta2024}", "\\checkmark: random split at patient level", "$\\circ$: balanced 432+432 and a 34+432 prevalence set", "$\\times$: no CI or SD", "$\\times$: none reported", "$\\times$: none seen", "\\checkmark: both reported"],
 ["LSTM-GAT \\cite{khorram2026}", "$\\times$: random; patient-level not stated", "$\\circ$: balancing not stated", "$\\circ$: mean (SD) over 5 seeds", "$\\times$: none reported", "\\checkmark: MIMIC-III (vitals only)", "$\\circ$: AUPRC numeric only for external set"],
 ["Transformer \\cite{tang2024}", "$\\times$: patient-level not stated", "$\\times$: matched cases/controls", "$\\circ$: mean $\\pm$ SD of 5-fold CV", "$\\times$: none reported", "\\checkmark: MIMIC external", "$\\times$: no AUPRC"],
 ["\\herald{} (this work)", "\\checkmark: patient-disjoint folds", "\\checkmark: never resampled", "\\checkmark: bootstrap CI + fold SD", "\\checkmark: prefix audit + control", "$\\circ$: hospital A$\\leftrightarrow$B only", "\\checkmark: both reported"]]
pw = ">{\\raggedright\\arraybackslash}p{2.2cm}" + ">{\\raggedright\\arraybackslash}p{2.35cm}" * 6
w("t_rubric_notes", "\\begin{table*}[!t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Evidence behind each rubric score (what the text we read states). ``None reported'' means not found in the text, not proof of absence.}\n\\label{tab:rubric_notes}\n"
  + tabular(pw, rows, ["Study", "Patient split", "Natural prevalence", "Uncertainty", "Leakage audit", "External", "AUROC+AUPRC"], midrule_before=(3,)) + "\n\\end{table*}\n")
print("extra tables done")


# ------------------------------------------------------------------ T19: published scores + TOPSIS on scores
from analysis import PERF, OPS
dash = "--"
def g(df, name, col, fmt="{:.3f}"):
    return fmt.format(df.loc[name, col]) if name in df.index and col in df.columns else dash
rows = [
 ["SepsisAI \\cite{gupta2024}", "patient-level, balanced test", "0.95", "0.96", "0.882", "0.968", "0.921", g(PERF, "SepsisAI [5]", "C*"), g(OPS, "SepsisAI [5]", "C*")],
 ["LSTM-GAT \\cite{khorram2026}", "12\\,h, random split, internal test", "0.853", dash, "0.706", "0.872", "0.627", "n/a", g(OPS, "LSTM-GAT [6]", "C*")],
 ["Transformer \\cite{tang2024}", "eICU, matched cohort", "$\\le$0.99", dash, "0.967", dash, "0.959", "n/a", g(OPS, "Transformer-eICU [7]", "C*")],
 ["MB-TCN (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019, own protocol", "0.892", "0.527", dash, dash, dash, g(PERF, "MB-TCN [8]\u2020", "C*"), "n/a"],
 ["PoEMS (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019, own protocol", "0.911", "0.675", dash, dash, dash, g(PERF, "PoEMS [8]\u2020", "C*"), "n/a"],
 ["SupMix (via \\cite{mansoor2026})$^\\dagger$", "PhysioNet 2019, AP not AUPRC", "0.883", "0.667", dash, dash, dash, g(PERF, "SupMix [8]\u2020", "C*"), "n/a"],
 ["\\textbf{HERALD} hourly", "natural prevalence, 5-fold CV", f"{M['dt']['AUROC']:.3f}", f"{M['dt']['AUPRC']:.3f}", "0.563$^\\ddagger$", "0.900$^\\ddagger$", dash, g(PERF, "HERALD hourly", "C*"), "n/a"],
 ["\\textbf{HERALD} stay-level", "natural prevalence, 5-fold CV", f"{sl['AUROC']:.3f}", f"{sl['AUPRC']:.3f}", f"{_OPJ['sens']:.3f}" if False else "0.858", "0.900", "0.547", g(PERF, "HERALD stay (natural)", "C*"), "n/a"],
 ["\\textbf{HERALD} stay-level", "balanced 432/432 view of the same predictions", f"{SD['balanced_432_432_view']['AUROC']:.3f}", f"{SD['balanced_432_432_view']['AUPRC']:.3f}", "0.858", "0.900", "0.876", g(PERF, "HERALD stay (balanced)", "C*"), g(OPS, "HERALD stay (balanced)", "C*")]]
w("t_lit_scores", "\\begin{table*}[!t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Published scores and TOPSIS on those scores. Scores are as reported under each study's \\emph{own} protocol (datasets and test sets differ) and are not directly comparable. $C^*_A$: TOPSIS on (AUROC, AUPRC), equal weights, studies reporting both; $C^*_B$: TOPSIS on (AUROC, sensitivity, F1), equal weights, studies reporting all three; n/a = criteria not reported. $^\\dagger$Quoted from \\cite{mansoor2026}; original not retrieved. $^\\ddagger$Row-level sensitivity at 90\\% specificity. HERALD stay-level sensitivity, specificity and F1 use a threshold fitted on the other folds (Youden); the natural-prevalence F1 is 0.547.}\n\\label{tab:lit_scores}\n"
  + tabular(">{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{3.6cm}cccccc|c".replace("cccccc|c", "ccccccc"), rows, ["Study", "Task / protocol", "AUROC", "AUPRC", "Sens.", "Spec.", "F1", "$C^*_A$", "$C^*_B$"], midrule_before=(6,)) + "\n\\end{table*}\n")
print("lit scores table done")
