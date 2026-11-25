import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from analysis import *
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve
from sklearn.isotonic import IsotonicRegression

GC = {"value": C["indigo"], "roll": C["sky"], "timing": C["amber"], "composite": C["emerald"], "static": C["slate"], "care": C["rose"]}
GNAME = {"value": "Last observed value", "roll": "Trailing / expanding statistics", "timing": "Measurement timing", "composite": "Clinical composites", "static": "Age, sex", "care": "Care pathway (ICULOS, unit, admission)"}
G_OF = {f: g for g, fs in COLS["groups"].items() for f in fs}
STAY = pd.read_parquet(f"{R}/results/stay_level_oof.parquet")
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC


def pretty(f):
    for a, b in (("_cnt", " · count"), ("_stale", " · staleness"), ("_rate", " · rate"), ("_xmax", " · stay max"), ("_xmin", " · stay min")):
        f = f.replace(a, b)
    return f


# ---------------------------------------------------------------- Fig 6: per-fold stability
def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw=dict(wspace=0.25))
    rows = []
    for lab, key, t in (("Early prediction 6 h", "cv/forecast", "forecast"), ("Hourly detection", "cv/detect", "detect")):
        pg = F2(key)["per_group"]; rows.append((lab, [pg[str(k)]["AUROC"] for k in range(5)], [pg[str(k)]["AUPRC"] / pg[str(k)]["prevalence"] for k in range(5)], t))
    au, lf = [], []
    for k in range(5):
        m = STAY[STAY.fold == k]; au.append(AUC(m.y, m.z_cv)); lf.append(AP(m.y, m.z_cv) / m.y.mean())
    rows.append(("Stay-level detection", au, lf, "stay"))
    rng = np.random.default_rng(1)
    for ai, (idx, ttl, yl) in enumerate(((1, "(a) AUROC per fold", "AUROC"), (2, "(b) AUPRC lift over prevalence per fold", "AUPRC / prevalence"))):
        ax = axs[ai]
        for yi, r in enumerate(rows):
            v = np.array(r[idx]); jitter = rng.uniform(-0.12, 0.12, len(v)); col = TASK_COL[r[3]]
            ax.scatter(v, np.full(len(v), yi) + jitter, s=46, color=col, alpha=0.85, edgecolor="white", linewidth=0.9, zorder=3)
            ax.hlines(yi, v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1), color=col, lw=5, alpha=0.28, zorder=2); ax.scatter(v.mean(), yi, marker="|", s=420, color=C["ink"], linewidth=1.6, zorder=4)
            ax.text(v.max() + (0.004 if ai == 0 else 0.25), yi + 0.2, f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if ai == 0 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}", fontsize=6.6, color=col, fontweight="bold")
        ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows] if ai == 0 else []); ax.set_xlabel(yl); ax.set_title(ttl, fontsize=8.2); ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, 2.7)
    axs[0].set_xlim(0.80, 0.99); axs[1].set_xlim(3, 14.5)
    save(fig, "fig06_folds")


# ---------------------------------------------------------------- Fig 10: importance + donut
def fig10():
    fig = plt.figure(figsize=(W2, 3.05)); gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.9], wspace=0.62)
    shares = None
    for i, (task, ttl) in enumerate((("forecast", "(a) Early prediction 6 h"), ("detect", "(b) Hourly detection"))):
        runs = J(f"cv_{task}_full.json"); acc = {}
        for r in runs:
            for f, v in r["top_features_gain"].items(): acc[f] = acc.get(f, 0) + v / len(runs)
        s = pd.Series(acc).sort_values(ascending=False); tot = s.sum(); top = (s / tot).head(11)[::-1]
        ax = fig.add_subplot(gs[0, i]); ax.barh(range(len(top)), top.values * 100, color=[GC[G_OF.get(f, "value")] for f in top.index], height=0.68, alpha=0.95)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([pretty(f) for f in top.index], fontsize=6.6); ax.set_xlabel("Share of top-15 gain (%)"); ax.set_title(ttl, fontsize=8.2); ax.grid(axis="y", visible=False)
        for yi, v in enumerate(top.values * 100): ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=6.2)
        if task == "forecast":
            shares = {}
            for f, v in (s / tot).items(): shares[G_OF.get(f, "value")] = shares.get(G_OF.get(f, "value"), 0) + v
    ax = fig.add_subplot(gs[0, 2]); order = sorted(shares, key=shares.get, reverse=True)
    w, _ = ax.pie([shares[g] for g in order], colors=[GC[g] for g in order], startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.6))
    ax.text(0, 0.06, "gain\nshare", ha="center", va="center", fontsize=7.5, fontweight="bold"); ax.set_title("(c) By feature group", fontsize=8.2)
    ax.legend(w, [f"{GNAME[g]}  {100 * shares[g]:.0f}%" for g in order], loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=5.9, ncol=1, handlelength=1.0)
    save(fig, "fig10_importance")


# ---------------------------------------------------------------- Fig 11: calibration
def crossfold_cal(task):
    d = pd.read_parquet(f"{R}/results/final_cv_{task}_preds.parquet", columns=["y", "z2", "fold"]); p = np.zeros(len(d))
    for k in range(5):
        te = (d.fold == k).to_numpy(); p[te] = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(d.z2[~te], d.y[~te]).predict(d.z2[te])
    d["p"] = p; return d


def fig11():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 0.8]))
    for ai, (task, col, lab) in enumerate((("forecast", C["indigo"], "Early prediction 6 h"), ("detect", C["teal"], "Hourly detection"))):
        d = crossfold_cal(task); edges = np.quantile(d.z2, [0, .5, .75, .9, .95, .98, .99, .995, 1.0]); b = np.clip(np.digitize(d.z2, edges[1:-1]), 0, 7)
        g = d.groupby(b).agg(pm=("p", "mean"), ym=("y", "mean"), n=("y", "size")); ax = axs[ai]
        ax.plot([1e-3, 1], [1e-3, 1], color=C["slate"], ls=":", lw=1); ax.plot(g.pm, g.ym, color=col, lw=1.6, alpha=0.8); ax.scatter(g.pm, g.ym, s=np.sqrt(g.n) / 3.2, color=col, edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e-3, 0.5); ax.set_ylim(2e-3, 0.5); ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed event rate" if ai == 0 else ""); ax.set_title(f"({'ab'[ai]}) {lab}", fontsize=8.2)
        ax.text(0.04, 0.9, f"ECE {F2('cv/' + task)['calibration_crossfold_isotonic']['ECE']:.4f}", transform=ax.transAxes, fontsize=7, color=col, fontweight="bold")
    ax = axs[2]; sk = [(100 * (1 - F2(f"cv/{t}")["calibration_crossfold_isotonic"]["Brier"] / F2(f"cv/{t}")["calibration_crossfold_isotonic"]["Brier_constant_prevalence"]), c, l) for t, c, l in (("forecast", C["indigo"], "Early\nprediction"), ("detect", C["teal"], "Hourly\ndetection"))]
    ax.bar(range(2), [s[0] for s in sk], color=[s[1] for s in sk], width=0.55, alpha=0.93)
    for i_, s in enumerate(sk): ax.text(i_, s[0] + 0.15, f"+{s[0]:.1f}%", ha="center", fontsize=7.5, fontweight="bold", color=s[1])
    ax.set_xticks(range(2)); ax.set_xticklabels([s[2] for s in sk]); ax.set_ylabel("Brier skill vs. constant risk (%)"); ax.set_title("(c) Brier skill", fontsize=8.2); ax.grid(axis="x", visible=False); ax.set_ylim(0, 8.5)
    save(fig, "fig11_calibration")


# ---------------------------------------------------------------- Fig 12: lead time at fixed false-alert budget
def fig12():
    d = pd.read_parquet(f"{R}/results/final_cv_forecast_preds.parquet", columns=["patient_id", "ICULOS", "z2"]).sort_values(["patient_id", "ICULOS"])
    ever = FOLDS.set_index("patient_id").ever_septic; d["ever"] = d.patient_id.map(ever).to_numpy(); g = d.groupby("patient_id").z2
    mx = g.max(); mx_ns = mx[~ever.reindex(mx.index).to_numpy().astype(bool)]
    sept = d[d.ever == True].groupby("patient_id").z2.apply(lambda s: s.to_numpy())
    fig, ax = plt.subplots(figsize=(W1, 2.7)); H = np.arange(0, 49)
    for bud, col in ((0.10, C["teal"]), (0.20, C["indigo"]), (0.30, C["coral"])):
        thr = np.quantile(mx_ns, 1 - bud); lead = []
        for z in sept:
            idx = np.flatnonzero(z >= thr); lead.append(len(z) - idx[0] if len(idx) else 0)
        lead = np.array(lead); cdf = [(lead >= h).mean() * 100 if h > 0 else (lead > 0).mean() * 100 for h in H]
        ax.plot(H, cdf, color=col, lw=2.1, label=f"{int(bud * 100)}% of non-septic stays alerted")
        ax.fill_between(H, 0, cdf, color=col, alpha=0.06); ax.scatter([6], [cdf[6]], s=36, color=col, edgecolor="white", zorder=4, linewidth=1)
        ax.text(6.6, cdf[6] + 2.2, f"{cdf[6]:.0f}%", fontsize=6.8, color=col, fontweight="bold")
    ax.axvline(6, color=C["slate"], ls=":", lw=0.9); ax.set_xlabel("Lead time: hours before first positive label"); ax.set_ylabel("Septic stays alerted ≥ this early (%)"); ax.set_xlim(0, 48); ax.set_ylim(0, 100); ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig12_leadtime")


# ---------------------------------------------------------------- Fig 13: stay-level
def fig13():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7), gridspec_kw=dict(wspace=0.36, width_ratios=[1, 1.05, 1.05]))
    ax = axs[0]
    for zc, col, lab in (("z_cv", C["coral"], "5-fold CV"), ("z_transfer", C["slate"], "hospital A↔B")):
        f, t, _ = roc_curve(STAY.y, STAY[zc]); ax.plot(f, t, color=col, lw=2, label=f"{lab} ({AUC(STAY.y, STAY[zc]):.3f})"); ax.fill_between(f, 0, t, color=col, alpha=0.06)
    ax.plot([0, 1], [0, 1], color=C["slate"], ls=":", lw=0.9); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.legend(loc="lower right", fontsize=6.4); ax.set_title("(a) Stay-level ROC", fontsize=8.2)
    ax = axs[1]; q = SD["AUROC_within_length_quartiles"]; names = list(q.keys()); vals = [q[k]["AUROC"] for k in names]
    ax.bar(range(4), vals, color=[C["coral"], "#FB923C", "#FDBA74", C["amber"]], width=0.62, alpha=0.95)
    for i_, v in enumerate(vals): ax.text(i_, v + 0.006, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.axhline(SD["AUROC_from_stay_length_alone"], color=C["rose"], ls="--", lw=1.2); ax.text(3.45, SD["AUROC_from_stay_length_alone"] + 0.012, f"length alone {SD['AUROC_from_stay_length_alone']:.2f}", ha="right", fontsize=6.6, color=C["rose"], fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"Q{i + 1}\n{k.split('[')[1].replace(']', '').replace(',', '–')}" for i, k in enumerate(names)], fontsize=5.9); ax.set_xlabel("stay length (h)", fontsize=6.8); ax.set_ylim(0.45, 1.0); ax.set_ylabel("AUROC within stay-length quartile"); ax.set_title("(b) Not a length shortcut", fontsize=8.2); ax.grid(axis="x", visible=False)
    ax = axs[2]; pr = SL["primary_no_length"]["cv_pooled"]; bal = SD["balanced_432_432_view"]; wl = SL["ablation_with_length_and_care"]["cv_pooled"]
    cats = [("Real\nprevalence", pr["AUROC"], pr["AUPRC"]), ("Balanced\n432/432", bal["AUROC"], bal["AUPRC"]), ("+ length &\ncare feats.", wl["AUROC"], wl["AUPRC"])]
    x = np.arange(3); ax.bar(x - 0.19, [c[1] for c in cats], 0.36, color=C["coral"], label="AUROC", alpha=0.95); ax.bar(x + 0.19, [c[2] for c in cats], 0.36, color=C["violet"], label="AUPRC", alpha=0.95)
    for i_, c in enumerate(cats): ax.text(i_ - 0.19, c[1] + 0.008, f"{c[1]:.2f}", ha="center", fontsize=6.3, fontweight="bold"); ax.text(i_ + 0.19, c[2] + 0.008, f"{c[2]:.2f}", ha="center", fontsize=6.3, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cats], fontsize=6.4); ax.set_ylim(0, 1.3); ax.legend(loc="upper center", fontsize=6.3, ncol=2); ax.set_title("(c) Protocol sensitivity", fontsize=8.2); ax.grid(axis="x", visible=False)
    save(fig, "fig13_stay_level")


# ---------------------------------------------------------------- Fig 14: own-model TOPSIS + radar
def fig14():
    fig = plt.figure(figsize=(W2, 3.0)); gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0]); c = pd.Series(TOP["equal"]).sort_values(); cols = [C["indigo"] if "HERALD" in n else "#CBD5E1" for n in c.index]
    ax.barh(range(len(c)), c.values, color=cols, height=0.66);
    for yi, v in enumerate(c.values): ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.2, fontweight="bold", color=C["indigo"] if "HERALD" in c.index[yi] else C["slate"])
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index); ax.set_xlim(0, 1.12); ax.set_xlabel("TOPSIS closeness  C*  (equal weights)"); ax.grid(axis="y", visible=False); panel(ax, "(a)", -0.45, 1.04)
    ax = fig.add_subplot(gs[0, 1], projection="polar"); cr = TOPC.copy(); cr["AlertsPerTP"] = -cr["AlertsPerTP"]; nz = (cr - cr.min()) / (cr.max() - cr.min()); labs = ["AUROC", "AUPRC", "Patient\nAUROC", "Sens@90", "Alerts/TP\n(inverted)"]
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist(); ang += ang[:1]
    for name, col in (("HERALD (this work)", C["indigo"]), ("LightGBM", C["coral"]), ("Causal encoder", C["slate"])):
        v = nz.loc[name].tolist(); v += v[:1]; ax.plot(ang, v, color=col, lw=1.9, label=name); ax.fill(ang, v, color=col, alpha=0.10)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, fontsize=6.8); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.05); ax.grid(color="#CBD5E1", lw=0.6); ax.spines["polar"].set_color("#CBD5E1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=6.5, ncol=1); ax.set_title("(b) Min–max normalized criteria", fontsize=8.2, pad=14)
    save(fig, "fig14_topsis_models")


# ---------------------------------------------------------------- Fig 15: weight-sensitivity heatmap
def fig15():
    M = pd.DataFrame(TOP).loc[["HERALD (this work)", "LightGBM", "LightGBM-reg", "LightGBM no-meas.", "Causal encoder", "Causal enc.+aux"]]
    fig, ax = plt.subplots(figsize=(W1, 2.6)); cm = LinearSegmentedColormap.from_list("t", ["#F8FAFC", "#99F6E4", C["teal"], "#134E4A"]); ax.grid(False)
    im = ax.imshow(M.values, cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M.values[i, j] > 0.55 else C["ink"], fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(["equal", "AUROC-\nheavy", "AUPRC-\nheavy", "op.-point-\nheavy"], fontsize=6.6); ax.set_yticks(range(6)); ax.set_yticklabels(M.index, fontsize=6.8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("C*", fontsize=7); cb.outline.set_visible(False)
    save(fig, "fig15_topsis_sensitivity")


# ---------------------------------------------------------------- Fig 16: literature rigor scorecard + weight sweep
def fig16():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1]))
    ax = axs[0]; S = SCORES[CRIT_FLAT]; cm = LinearSegmentedColormap.from_list("r", ["#FDA4AF", "#FDE68A", "#6EE7B7"]); ax.grid(False)
    ax.imshow(S.values, cmap=cm, vmin=0, vmax=1, aspect="auto")
    sym = {0: "✗", 0.5: "◐", 1: "✓"}
    for i in range(S.shape[0]):
        for j in range(S.shape[1]): ax.text(j, i, sym[S.values[i, j]], ha="center", va="center", fontsize=11, color=C["ink"])
        ax.text(S.shape[1] - 0.35, i, f"{SCORES['Rigor score'].iloc[i]:.2f}", ha="left", va="center", fontsize=8, fontweight="bold", color=C["indigo"] if i < 2 else C["slate"])
    ax.text(S.shape[1] - 0.35, -0.72, "Score", fontsize=7, fontweight="bold", color=C["ink"])
    SHORT = ["Patient\nsplit", "Natural\nprev.", "Uncert.\nreported", "Leakage\naudit", "External\nsite", "AUROC\n+AUPRC"]; ax.set_xticks(range(6)); ax.set_xticklabels(SHORT, fontsize=5.8); ax.xaxis.tick_top(); ax.set_yticks(range(5)); ax.set_yticklabels(PAPERS, fontsize=7); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(-0.5, 6.4); panel(ax, "(a)", -0.38, 1.16)
    ax = axs[1]; cols = [C["indigo"], C["violet"], C["coral"], C["teal"], C["amber"]]
    for p, c_ in zip(PAPERS, cols): ax.plot(LIT_SWEEP.index, LIT_SWEEP[p], color=c_, lw=2.2 if "HERALD" in p else 1.6, ls="-" if "HERALD" in p else "--", label=p)
    ax.axvline(0.5, color=C["slate"], ls=":", lw=0.9); ax.set_xlabel("Weight on rigor  (1 − weight on AUROC)"); ax.set_ylabel("TOPSIS closeness  C*"); ax.set_ylim(-0.02, 1.05); ax.legend(loc="upper center", fontsize=6.0, bbox_to_anchor=(0.5, -0.2), ncol=2); panel(ax, "(b)", -0.16, 1.16)
    save(fig, "fig16_lit_topsis")


# ---------------------------------------------------------------- Fig 17: negative results (validation-only)
def fig17():
    rows = []
    for task, tl, col in (("forecast", "Early prediction", C["indigo"]), ("detect", "Detection", C["teal"])):
        base = R3[f"{task}/E0_v2_features"]["val_AUROC"]
        for k, l in (("E1_plus_personal_baseline", "Personal-baseline features"), ("E2_multihorizon_soft_label", "Multi-horizon soft label"), ("E3_pos_weight3", "Positive-class weight ×3"), ("E4_avg_of_3_configs", "Average of 3 configurations")):
            if f"{task}/{k}" in R3: rows.append((l, tl, R3[f"{task}/{k}"]["val_AUROC"] - base, col))
        rows.append(("306 → 370 features (adopted)", tl, F12[f"X2.npy/{task}"]["val_AUROC"] - F12[f"X.npy/{task}"]["val_AUROC"], col))
    df = pd.DataFrame(rows, columns=["idea", "task", "d", "col"]); order = ["306 → 370 features (adopted)", "Personal-baseline features", "Multi-horizon soft label", "Average of 3 configurations", "Positive-class weight ×3"][::-1]
    fig, ax = plt.subplots(figsize=(W1 + 0.5, 2.8)); h = 0.34
    for ti, (tl, col) in enumerate((("Early prediction", C["indigo"]), ("Detection", C["teal"]))):
        for yi, idea in enumerate(order):
            m = df[(df.idea == idea) & (df.task == tl)]
            if len(m): v = m.d.iloc[0] * 1000; ax.barh(yi + (ti - 0.5) * h * 1.05, v, height=h, color=col, alpha=0.93, label=tl if yi == 0 else None); ax.text(v + (0.5 if v >= 0 else -0.5), yi + (ti - 0.5) * h * 1.05, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=6.4)
    ax.axvline(0, color=C["ink"], lw=0.9); ax.axvline(1.0, color=C["rose"], lw=1.0, ls="--"); ax.text(1.3, 1.55, "adoption\nthreshold\n+1.0", fontsize=5.8, color=C["rose"], va="center")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=6.8); ax.set_xlabel("Δ validation AUROC  (×10⁻³)"); ax.set_xlim(-26, 8); ax.grid(axis="y", visible=False); ax.legend(loc="upper left", fontsize=6.6)
    save(fig, "fig17_negative_results")


# ---------------------------------------------------------------- Fig 18: model family comparison
def fig18():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw=dict(wspace=0.22))
    fam = [("Boosting\n(1 stage)", C["teal"]), ("TCN", C["sky"]), ("Boosting +\nTCN blend", C["violet"]), ("Two-stage\n(final)", C["indigo"])]
    for ai, (met, ttl) in enumerate((("AUROC", "(a) AUROC"), ("AUPRC", "(b) AUPRC"))):
        ax = axs[ai]
        for ti, (task, tl) in enumerate((("forecast", "Early prediction 6 h"), ("detect", "Hourly detection"))):
            g = GB[f"cv/{task}"]; v = [g["GBM"]["pooled"][met], g["TCN"]["pooled"][met], g["ENSEMBLE"]["pooled"][met], F2(f"cv/{task}")["pooled"][met]]
            for k, (val, (l, c_)) in enumerate(zip(v, fam)):
                x = ti + (k - 1.5) * 0.2; ax.bar(x, val, 0.18, color=c_, alpha=0.95, label=l.replace("\n", " ") if ti == 0 else None); ax.text(x, val + (0.004 if met == "AUROC" else 0.003), f"{val:.3f}", ha="center", fontsize=5.4, fontweight="bold", rotation=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Early prediction 6 h", "Hourly detection"]); ax.set_title(ttl, fontsize=8.2); ax.grid(axis="x", visible=False)
        ax.set_ylim((0.74, 0.90) if met == "AUROC" else (0, 0.17))
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=7)
    save(fig, "fig18_model_family")


# ---------------------------------------------------------------- Fig 19: performance vs rigor bubble
def fig19():
    fig, ax = plt.subplots(figsize=(W1, 2.8)); cols = [C["indigo"], C["violet"], C["coral"], C["teal"], C["amber"]]
    for p, c_ in zip(PAPERS, cols):
        x, y = SCORES.loc[p, "Rigor score"], AUROC[p]; ax.scatter(x, y, s=330 if "HERALD" in p else 230, color=c_, alpha=0.85, edgecolor="white", linewidth=1.6, zorder=3)
        ax.annotate(p, (x, y), xytext=(8, -3 if "stay" not in p else 7), textcoords="offset points", fontsize=6.5, fontweight="bold", color=c_)
    ax.set_xlim(0.1, 1.12); ax.set_ylim(0.82, 1.02); ax.set_xlabel("Rigor score (reporting checklist, 0–1)"); ax.set_ylabel("Headline AUROC (own protocol)")
    ax.axhspan(0.82, 0.86, color="#F1F5F9", zorder=0); ax.text(0.11, 0.826, "AUROCs are not directly comparable across protocols", fontsize=5.9, color=C["slate"], style="italic")
    save(fig, "fig19_perf_vs_rigor")


if __name__ == "__main__":
    for f in (fig6, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19):
        f()
