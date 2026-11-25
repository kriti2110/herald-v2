# HERALD-v2 results (exact, from results/final_metrics.json)

Real prevalence, no resampling of any test set. 95% CIs: patient-clustered bootstrap (1000 reps) on pooled out-of-fold predictions.


## Early prediction (onset within 6 h), 5-fold patient-level CV

| Model | AUROC [95% CI] | AUPRC [95% CI] | prevalence |
|---|---|---|---|
| GBM | 0.8235 [0.8156, 0.8310] | 0.0589 [0.0549, 0.0634] | 0.924% |
| TCN | 0.8028 [0.7950, 0.8111] | 0.0462 [0.0435, 0.0492] | 0.924% |
| ENSEMBLE | 0.8225 [0.8148, 0.8303] | 0.0555 [0.0521, 0.0595] | 0.924% |

## Detection (SepsisLabel at current hour), 5-fold patient-level CV

| Model | AUROC [95% CI] | AUPRC [95% CI] | prevalence |
|---|---|---|---|
| GBM | 0.8487 [0.8429, 0.8543] | 0.1302 [0.1220, 0.1390] | 1.798% |
| TCN | 0.8201 [0.8139, 0.8263] | 0.1076 [0.1010, 0.1151] | 1.798% |
| ENSEMBLE | 0.8430 [0.8369, 0.8487] | 0.1308 [0.1231, 0.1397] | 1.798% |

## Early prediction, hospital transfer (A->B and B->A pooled)

| Model | AUROC [95% CI] | AUPRC [95% CI] | prevalence |
|---|---|---|---|
| GBM | 0.7579 [0.7485, 0.7675] | 0.0380 [0.0354, 0.0409] | 0.924% |
| TCN | 0.7531 [0.7439, 0.7623] | 0.0391 [0.0367, 0.0422] | 0.924% |
| ENSEMBLE | 0.7672 [0.7574, 0.7768] | 0.0410 [0.0381, 0.0443] | 0.924% |

## Detection, hospital transfer (A->B and B->A pooled)

| Model | AUROC [95% CI] | AUPRC [95% CI] | prevalence |
|---|---|---|---|
| GBM | 0.7959 [0.7890, 0.8032] | 0.0884 [0.0827, 0.0948] | 1.798% |
| TCN | 0.7688 [0.7608, 0.7772] | 0.0758 [0.0716, 0.0809] | 1.798% |
| ENSEMBLE | 0.7957 [0.7890, 0.8031] | 0.0926 [0.0866, 0.0997] | 1.798% |

## Ablations (GBM, pooled)

| Setting | Variant | AUROC | AUPRC |
|---|---|---|---|
| cv/forecast | no_care | 0.8182 | 0.0568 |
| cv/forecast | physio_only | 0.7980 | 0.0476 |
| cv/forecast | full | 0.8235 | 0.0589 |
| transfer/forecast | physio_only | 0.7165 | 0.0353 |
| transfer/forecast | full | 0.7579 | 0.0380 |
| cv/detect | no_care | 0.8427 | 0.1281 |
| cv/detect | physio_only | 0.8222 | 0.1031 |
| cv/detect | full | 0.8487 | 0.1302 |
| transfer/detect | physio_only | 0.7625 | 0.0740 |
| transfer/detect | full | 0.7959 | 0.0884 |

## Audits
- Pipeline prefix-invariance: PASS, 306/306 features, 1,688 cuts, 200 stays; leaky control caught (1,412 violations).
- TCN prefix check: max |logit diff| 4.8e-07.


## Protocol notes
- Ensemble = 0.5/0.5 logit average, fixed before seeing test results.
- Hyper-parameters chosen on validation patients only (scripts/04_tune.py).
- A first run used a faulty early-stopping call (stopped on any metric); it was discarded (results/_discarded_auc_stopping) and re-run after the fix.


# FINAL two-stage model (v2 features + temporal stacking) — exact values

Real prevalence, no resampling; CI = patient-clustered bootstrap (1000 reps). Source: results/final_twostage_metrics.json

| Setting | Model | AUROC [95% CI] | AUPRC [95% CI] | patient-level AUROC / AUPRC | balanced-patient view AUROC / AUPRC |
|---|---|---|---|---|---|
| cv/forecast | stage1 | 0.8253 [0.8175, 0.8327] | 0.0600 [0.0554, 0.0653] | 0.7971 / 0.4403 | 0.7967 / 0.8279 |
| cv/forecast | FINAL_stage2 | 0.8240 [0.8167, 0.8318] | 0.0548 [0.0513, 0.0594] | 0.7952 / 0.4220 | 0.7962 / 0.8266 |
| cv/detect | stage1 | 0.8502 [0.8449, 0.8565] | 0.1319 [0.1236, 0.1407] | 0.8691 / 0.5106 | 0.8694 / 0.8790 |
| cv/detect | FINAL_stage2 | 0.8519 [0.8464, 0.8580] | 0.1342 [0.1263, 0.1426] | 0.8705 / 0.5187 | 0.8703 / 0.8801 |
| transfer/forecast | stage1 | 0.7689 [0.7598, 0.7779] | 0.0392 [0.0366, 0.0423] | 0.6617 / 0.3004 | 0.6614 / 0.7343 |
| transfer/forecast | FINAL_stage2 | 0.7727 [0.7637, 0.7817] | 0.0406 [0.0380, 0.0440] | 0.7059 / 0.3211 | 0.7039 / 0.7582 |
| transfer/detect | stage1 | 0.8002 [0.7932, 0.8074] | 0.0931 [0.0873, 0.0995] | 0.8086 / 0.4073 | 0.8097 / 0.8276 |
| transfer/detect | FINAL_stage2 | 0.7953 [0.7883, 0.8022] | 0.0877 [0.0821, 0.0941] | 0.7958 / 0.3780 | 0.7954 / 0.8139 |

The temporal stage-2 added no meaningful gain (forecast: 0.8253->0.8240 AUROC; detect: 0.8502->0.8519). Stage 1 and stage 2 both reported; FINAL_stage2 was the pre-specified model.
