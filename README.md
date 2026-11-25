# HERALD-v2: prefix-valid, audited early prediction and detection of sepsis

Code, fold assignments, saved out-of-fold predictions and the scripts that regenerate every table and figure for the paper
*HERALD: Prefix-Valid, Audited Early Prediction and Detection of Sepsis with Patient-Level and Cross-Hospital Evaluation on PhysioNet/CinC 2019*.

## Data
The PhysioNet/CinC 2019 challenge data (training sets A and B) are **not redistributed here**. Download them from
<https://physionet.org/content/challenge-2019/> and place `training_setA/` and `training_setB/` in the repository root
(or point `01_build_features.py` at them). Raw data are converted once to `cache/raw_long.parquet`; that file and the large feature matrices `cache/X*.npy` are not tracked (the small `cache/columns*.json` and `meta.parquet` are). `results/patient_attributes.csv` holds per-patient age and sex so `16_subgroups_dca_burden.py` runs from the saved predictions alone.

## Layout
| Path | Contents |
|---|---|
| `hv2/` | strictly causal feature builder (`features.py`) and causal temporal CNN (`tcn.py`) |
| `scripts/01*_build_features*.py` | build the causal features (v1: 306, v3: 370 features) |
| `scripts/02*_audit_prefix*.py` | prefix-invariance audit with the centred-window negative control |
| `scripts/03_gbm.py`, `04_tune.py`, `05_tcn.py`, `07_final.py` | models, validation-only tuning, final two-stage cross-validation and hospital transfer (`run_final.sh`) |
| `scripts/09`-`15` | validation-only experiments, horizons, stay-level task, TOPSIS on identical patients, stay-level OOF export |
| `scripts/16_subgroups_dca_burden.py` | subgroups (age, sex, hospital), decision curves, alert burden (reads saved predictions only) |
| `scripts/17_label_timing_sensitivity.py` | label-timing sensitivity of the early-prediction result (evaluation only, reads saved predictions) |
| `scripts/18_cohort_flow_counts.py` | cohort characteristics by hospital, participant-flow numbers, patients and events per training stage |
| `scripts/19_train_release_models.py`, `hv2/twostage.py` | trains the final two-stage models on all patients and saves them to `models/`; `hv2.twostage.predict` scores new data |
| `docs/FEATURES.md`, `configs/final_model.json` | definition of all 370 features; seeds, hyper-parameters, software versions |
| `models/` | trained LightGBM files (4 stage-1 models + 1 stage-2 model per task, trained on all patients; see `model_info.json`) |
| `results/` | fold assignments (`patient_folds.csv`), final out-of-fold predictions, metrics and audit JSON files |
| `paper/` | figure and table generators (`figs_*.py`, `tables.py`, `analysis.py`), `numbers.tex`, `main.tex`, supplementary code listings, graphical abstract |

## Reproducing
```bash
pip install -r requirements.txt
# 1. features + audit
python scripts/01c_build_features_v3.py && python scripts/02c_audit_prefix_v3.py
# 2. final models (5-fold patient-level CV, train A -> test B and B -> A)
bash scripts/run_final.sh forecast && bash scripts/run_final.sh detect
# 3. analyses, tables and figures from the saved result files
python scripts/16_subgroups_dca_burden.py && python scripts/17_label_timing_sensitivity.py
cd paper && python figs_a.py && python figs_b.py && python figs_c.py && python figs_d.py && python tables.py
```
Outer folds use seed 42. `HERALD_ROOT` overrides the repository root; `scripts/13_topsis_same_patients.py` also needs the outputs of the
earlier baseline project (`HERALD_V1_ROOT`), which are not included. The saved predictions in `results/` are enough to
regenerate the paper's tables and figures without retraining (run times for retraining: roughly 30-60 minutes per task on a 10-core laptop).

Scripts `01*`, `02*` and `18` need the raw PhysioNet data (see Data); scripts `16` and `17` and the paper figure/table generators run from the files in this repository alone.

## Using the trained models
```python
import numpy as np, pandas as pd
from hv2.features import build_features
from hv2.twostage import predict
meta, X, _ = build_features(raw_hourly_df, extra=True)       # raw_hourly_df: PhysioNet-format hourly rows (patient_id, ICULOS, vitals, labs, ...)
codes, _ = pd.factorize(meta["patient_id"])
z = predict("models/forecast", X, codes, meta["ICULOS"].to_numpy())   # raw logit for 6 h early prediction; use models/detect for hourly detection
risk = 1 / (1 + np.exp(-z))                                      # uncalibrated probability; recalibrate locally before any use
```
The released models are trained on all 40,336 patients, so their in-sample performance is optimistic; the reported performance comes from the patient-level cross-validation in `results/`. They are research artefacts, not clinical tools, and thresholds must be validated locally.

## Notes
* Predictions in `results/*_preds.parquet` are out-of-fold (or cross-hospital) scores: `z1` stage-1 logit, `z2` final two-stage logit.
* Only the four final prediction files are included; intermediate and ablation predictions are regenerated by `07_final.py` / `03_gbm.py`.
* Code is released under the MIT License (see `LICENSE`). The PhysioNet/CinC 2019 data keep their own licence and are not included.
