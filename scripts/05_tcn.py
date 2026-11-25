"""Train the causal multi-task TCN (cv or hospital transfer); save test predictions
for BOTH tasks. Same patient folds and same fit/val/calib split as 03_gbm.py."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import train_test_split
from hv2.features import PID, CONT
from hv2.tcn import CausalMTL

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="cv", choices=["cv", "transfer"])
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
POS_W = 5.0

meta = pd.read_parquet(f"{R}/cache/meta.parquet")
info = json.load(open(f"{R}/cache/columns.json")); cols = info["columns"]; ci = {c: i for i, c in enumerate(cols)}
Xall = np.load(f"{R}/cache/X.npy", mmap_mode="r")
val_i = [ci[c] for c in CONT]; stale_i = [ci[f"{c}_stale"] for c in CONT]
stat_i = [ci[c] for c in ["Age", "Gender", "ICULOS", "HospAdmTime", "Unit1", "Unit2"]]
V = np.asarray(Xall[:, val_i]); S = np.asarray(Xall[:, stale_i]); ST = np.asarray(Xall[:, stat_i])
obs = (S == 0).astype(np.float32)
lstale = np.log1p(np.where(np.isnan(S), 72.0, np.minimum(S, 72.0))).astype(np.float32) / 4.3
pc, upid = pd.factorize(meta[PID])
n_pat = len(upid)
starts = np.r_[0, np.flatnonzero(np.diff(pc)) + 1]; ends = np.r_[starts[1:], len(pc)]
assert len(starts) == n_pat
fc_keep = meta["fc_keep"].to_numpy(); fc_t = meta["fc_target"].to_numpy(); det_t = meta["det_target"].to_numpy()
folds = pd.read_csv(f"{R}/results/patient_folds.csv").set_index(PID).loc[upid]
fold_of, ever, hosp = folds["fold"].to_numpy(), folds["ever_septic"].to_numpy(), folds["hospital"].to_numpy()


def make_inputs(fit_pat):
    rows = np.flatnonzero(np.isin(pc, fit_pat))
    mu, sd = np.nanmean(V[rows], 0), np.nanstd(V[rows], 0) + 1e-6
    smu, ssd = np.nanmean(ST[rows], 0), np.nanstd(ST[rows], 0) + 1e-6
    v = np.clip((V - mu) / sd, -6, 6); v = np.nan_to_num(v, nan=0.0)
    st = np.nan_to_num(np.clip((ST - smu) / ssd, -6, 6), nan=0.0)
    return np.concatenate([v, obs, lstale, st], 1).astype(np.float32)


def batch(Xin, pats, pad_to=None):
    L = [ends[p] - starts[p] for p in pats]; T = max(L)
    x = np.zeros((len(pats), T, Xin.shape[1]), np.float32); y = np.zeros((len(pats), T, 2), np.float32)
    m = np.zeros((len(pats), T, 2), np.float32)
    for i, p in enumerate(pats):
        s, e = starts[p], ends[p]; n = e - s
        x[i, :n] = Xin[s:e]; y[i, :n, 0] = fc_t[s:e]; y[i, :n, 1] = det_t[s:e]
        m[i, :n, 0] = fc_keep[s:e]; m[i, :n, 1] = 1.0
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.from_numpy(m).to(dev), L


@torch.no_grad()
def predict(model, Xin, pats, bs=128):
    model.eval(); out = np.full((len(pc), 2), np.nan, np.float32)
    order = sorted(pats, key=lambda p: ends[p] - starts[p])
    for i in range(0, len(order), bs):
        pp = order[i:i + bs]; x, _, _, L = batch(Xin, pp)
        z = model(x).float().cpu().numpy()
        for j, p in enumerate(pp):
            out[starts[p]:ends[p]] = z[j, :L[j]]
    return out


def val_score(z, pats):
    rows = np.concatenate([np.arange(starts[p], ends[p]) for p in pats])
    f = rows[fc_keep[rows]]
    return 0.5 * (AUC(fc_t[f], z[f, 0]) + AUC(det_t[rows], z[rows, 1])), \
           AP(fc_t[f], z[f, 0]), AP(det_t[rows], z[rows, 1])


def run(train_pat, test_pat, tag):
    t0 = time.time()
    if args.smoke:
        train_pat = np.random.default_rng(0).choice(train_pat, 2500, replace=False); test_pat = test_pat[:400]
    tr_i, va_i = train_test_split(np.arange(len(train_pat)), test_size=0.25, stratify=ever[train_pat], random_state=args.seed)
    tr_pat, va_pat = train_pat[tr_i], train_pat[va_i]
    va_pat = train_test_split(va_pat, test_size=0.5, stratify=ever[va_pat], random_state=args.seed)[0]
    Xin = make_inputs(tr_pat)
    model = CausalMTL(Xin.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr_pat) / 64)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    best, best_state, bad = -1, None, 0
    for ep in range(args.epochs):
        model.train(); perm = np.random.permutation(tr_pat); tl = 0
        # length-bucketed batches: shuffle, sort within chunks of 64*20
        chunks = [sorted(perm[i:i + 1280], key=lambda p: ends[p] - starts[p]) for i in range(0, len(perm), 1280)]
        bl = [c[i:i + 64] for c in chunks for i in range(0, len(c), 64)]; np.random.shuffle(bl)
        for pp in bl:
            x, y, m, _ = batch(Xin, pp)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, pos_weight=torch.tensor(POS_W, device=dev), reduction="none") * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
        zv = predict(model, Xin, list(va_pat)); sc, a1, a2 = val_score(zv, list(va_pat))
        print(f"[{tag}] ep{ep} loss {tl/len(bl):.4f} val AUROC(avg fc,det) {sc:.4f} AP fc {a1:.4f} det {a2:.4f} {time.time()-t0:.0f}s", flush=True)
        if sc > best: best, bad, best_state = sc, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3: break
    model.load_state_dict(best_state)
    z = predict(model, Xin, list(test_pat))
    rows = np.concatenate([np.arange(starts[p], ends[p]) for p in test_pat])
    out = pd.DataFrame({PID: upid[pc[rows]], "ICULOS": meta["ICULOS"].to_numpy()[rows],
                        "z_fc": z[rows, 0] - np.log(POS_W), "z_det": z[rows, 1] - np.log(POS_W),
                        "fc_keep": fc_keep[rows], "y_fc": fc_t[rows], "y_det": det_t[rows]})
    # prefix-validity audit of the trained network itself (inputs were audited separately)
    if tag in ("fold0", "trainA_testB", "smoke"):
        rng = np.random.default_rng(0); worst = 0.0
        for p in rng.choice(test_pat, 40, replace=False):
            s, e = starts[p], ends[p]; t = int(rng.integers(0, e - s))
            x_full, *_ = batch(Xin, [p]); 
            with torch.no_grad():
                a = model.eval()(x_full)[0, t].float().cpu().numpy()
                b = model.eval()(x_full[:, : t + 1])[0, t].float().cpu().numpy()
            worst = max(worst, float(np.abs(a - b).max()))
        json.dump({"tag": tag, "patients": 40, "max_abs_logit_diff_full_vs_prefix": worst, "PASS": worst < 1e-3},
                  open(f"{R}/results/tcn_prefix_audit_{tag}.json", "w"), indent=2)
        print(f"[{tag}] TCN prefix audit max |logit diff| = {worst:.2e}", flush=True)
    return out


name = f"{args.mode}_tcn" + ("_smoke" if args.smoke else "")
outs = []
allp = np.arange(n_pat)
if args.mode == "cv":
    for k in [int(f) for f in args.folds.split(",")]:
        o = run(allp[fold_of != k], allp[fold_of == k], "smoke" if args.smoke else f"fold{k}"); o["fold"] = k; outs.append(o)
else:
    for a, b in [("A", "B"), ("B", "A")]:
        o = run(np.flatnonzero(hosp == a), np.flatnonzero(hosp == b), f"train{a}_test{b}"); o["direction"] = f"{a}->{b}"; outs.append(o)
pd.concat(outs).to_parquet(f"{R}/results/{name}_preds.parquet"); print("saved", name)
