# The 370 causal features

Every feature at hour *t* is a function of that patient's rows 1..*t* only (strict prefix validity, Eq. 1 of the paper). Allowed operations: forward-fill (LOCF), trailing rolling windows ending at *t*, expanding statistics, positive shifts and cumulative sums. Forbidden: backward fill, centred or leading windows, per-stay statistics over the whole sequence, any scaler fitted on all data. `scripts/02*_audit_prefix*.py` verifies this end to end; the builder is `hv2/features.py` (`build_features(raw, extra=True)` gives these 370 columns).

The 34 continuous variables are the 8 vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2) and 26 laboratory values (BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets).

## Feature groups

| Group | Count | Description |
|---|---|---|
| `value` | 34 | Last observed value (forward-filled, LOCF) of each of the 34 vital-sign and laboratory variables. |
| `timing` | 104 | Measurement timing and frequency: hours since last measured (`_stale`), measurements so far (`_cnt`), measurement rate (`_rate`), labs ordered in the last 6 and 24 h (`n_labs6`, `n_labs24`). These are the charting-behaviour features removed in the physiology-only ablation. |
| `roll` | 191 | Trailing and expanding statistics and changes: trailing 3/6/12/24 h means, SD, min, max; deltas vs 6, 12, 24 h ago; slopes; expanding (stay-so-far) minima and maxima; change since the previous measurement (`_dobs`). |
| `composite` | 35 | Clinical composites and threshold flags from forward-filled values: shock index, pulse pressure, SIRS, qSOFA, an organ-dysfunction count, abnormal-hour counts over trailing 6/24 h, running maxima, time since first abnormality, BUN/creatinine, SaO2/FiO2. |
| `static` | 2 | Demographics: age and sex (coded as in the challenge data). |
| `care` | 4 | Care-pathway / workflow variables: ICU length of stay so far (`ICULOS`), hospital admission time (`HospAdmTime`), ICU unit indicators (`Unit1`, `Unit2`). Removed in the no-care ablation. |

Ablations used in the paper: *no care* drops the `care` group; *physiology-only* drops the `care` and `timing` groups.

## Naming conventions

| Pattern | Meaning |
|---|---|
| `X` | last observed value of variable X (LOCF) |
| `X_stale` | hours since X was last measured |
| `X_cnt` | number of measurements of X so far |
| `X_rate` | `X_cnt` divided by hours elapsed |
| `V_mean{3,6,12,24}`, `V_std{6,24}`, `V_min{6,24}`, `V_max{6,24}` | trailing-window statistics of vital V ending at hour t |
| `V_d{6,24}`, `L_d12` | value now minus value 6, 24 or 12 hours ago (L: seven key labs) |
| `V_slope6` | change in the trailing 6 h mean over the last 6 h, per hour |
| `X_xmin`, `X_xmax` | stay-so-far minimum and maximum |
| `X_dobs` | change since the previous *measurement* of X |
| `shock_index`, `pulse_pressure` | HR / SBP; SBP - DBP |
| `sirs`, `qsofa`, `organ_dys_count` | SIRS criteria met (0-4), qSOFA (0-2), count of six organ-dysfunction thresholds |
| `S_max6`, `S_max24`, `S_cummax` | trailing and running maxima of score S |
| `hr_hi_n6`, `rr_hi_n6`, ... | hours in the last 6 / 24 h with a threshold flag set (`_n6`, `_n24`) |
| `t_since_first_F` | hours since flag F (map_low, lactate_hi, sirs2) was first set |
| `bun_creat`, `sao2_fio2` | BUN / creatinine; SaO2 / FiO2 |

## Full list

### `value` (34)

`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, `EtCO2`, `BaseExcess`, `HCO3`, `FiO2`, `pH`, `PaCO2`, `SaO2`, `AST`, `BUN`, `Alkalinephos`, `Calcium`, `Chloride`, `Creatinine`, `Bilirubin_direct`, `Glucose`, `Lactate`, `Magnesium`, `Phosphate`, `Potassium`, `Bilirubin_total`, `TroponinI`, `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets`

### `timing` (104)

`HR_stale`, `HR_cnt`, `HR_rate`, `O2Sat_stale`, `O2Sat_cnt`, `O2Sat_rate`, `Temp_stale`, `Temp_cnt`, `Temp_rate`, `SBP_stale`, `SBP_cnt`, `SBP_rate`, `MAP_stale`, `MAP_cnt`, `MAP_rate`, `DBP_stale`, `DBP_cnt`, `DBP_rate`, `Resp_stale`, `Resp_cnt`, `Resp_rate`, `EtCO2_stale`, `EtCO2_cnt`, `EtCO2_rate`, `BaseExcess_stale`, `BaseExcess_cnt`, `BaseExcess_rate`, `HCO3_stale`, `HCO3_cnt`, `HCO3_rate`, `FiO2_stale`, `FiO2_cnt`, `FiO2_rate`, `pH_stale`, `pH_cnt`, `pH_rate`, `PaCO2_stale`, `PaCO2_cnt`, `PaCO2_rate`, `SaO2_stale`, `SaO2_cnt`, `SaO2_rate`, `AST_stale`, `AST_cnt`, `AST_rate`, `BUN_stale`, `BUN_cnt`, `BUN_rate`, `Alkalinephos_stale`, `Alkalinephos_cnt`, `Alkalinephos_rate`, `Calcium_stale`, `Calcium_cnt`, `Calcium_rate`, `Chloride_stale`, `Chloride_cnt`, `Chloride_rate`, `Creatinine_stale`, `Creatinine_cnt`, `Creatinine_rate`, `Bilirubin_direct_stale`, `Bilirubin_direct_cnt`, `Bilirubin_direct_rate`, `Glucose_stale`, `Glucose_cnt`, `Glucose_rate`, `Lactate_stale`, `Lactate_cnt`, `Lactate_rate`, `Magnesium_stale`, `Magnesium_cnt`, `Magnesium_rate`, `Phosphate_stale`, `Phosphate_cnt`, `Phosphate_rate`, `Potassium_stale`, `Potassium_cnt`, `Potassium_rate`, `Bilirubin_total_stale`, `Bilirubin_total_cnt`, `Bilirubin_total_rate`, `TroponinI_stale`, `TroponinI_cnt`, `TroponinI_rate`, `Hct_stale`, `Hct_cnt`, `Hct_rate`, `Hgb_stale`, `Hgb_cnt`, `Hgb_rate`, `PTT_stale`, `PTT_cnt`, `PTT_rate`, `WBC_stale`, `WBC_cnt`, `WBC_rate`, `Fibrinogen_stale`, `Fibrinogen_cnt`, `Fibrinogen_rate`, `Platelets_stale`, `Platelets_cnt`, `Platelets_rate`, `n_labs6`, `n_labs24`

### `roll` (191)

`HR_mean6`, `HR_std6`, `HR_min6`, `HR_max6`, `HR_d6`, `HR_mean24`, `HR_std24`, `HR_min24`, `HR_max24`, `HR_d24`, `O2Sat_mean6`, `O2Sat_std6`, `O2Sat_min6`, `O2Sat_max6`, `O2Sat_d6`, `O2Sat_mean24`, `O2Sat_std24`, `O2Sat_min24`, `O2Sat_max24`, `O2Sat_d24`, `Temp_mean6`, `Temp_std6`, `Temp_min6`, `Temp_max6`, `Temp_d6`, `Temp_mean24`, `Temp_std24`, `Temp_min24`, `Temp_max24`, `Temp_d24`, `SBP_mean6`, `SBP_std6`, `SBP_min6`, `SBP_max6`, `SBP_d6`, `SBP_mean24`, `SBP_std24`, `SBP_min24`, `SBP_max24`, `SBP_d24`, `MAP_mean6`, `MAP_std6`, `MAP_min6`, `MAP_max6`, `MAP_d6`, `MAP_mean24`, `MAP_std24`, `MAP_min24`, `MAP_max24`, `MAP_d24`, `DBP_mean6`, `DBP_std6`, `DBP_min6`, `DBP_max6`, `DBP_d6`, `DBP_mean24`, `DBP_std24`, `DBP_min24`, `DBP_max24`, `DBP_d24`, `Resp_mean6`, `Resp_std6`, `Resp_min6`, `Resp_max6`, `Resp_d6`, `Resp_mean24`, `Resp_std24`, `Resp_min24`, `Resp_max24`, `Resp_d24`, `EtCO2_mean6`, `EtCO2_std6`, `EtCO2_min6`, `EtCO2_max6`, `EtCO2_d6`, `EtCO2_mean24`, `EtCO2_std24`, `EtCO2_min24`, `EtCO2_max24`, `EtCO2_d24`, `HR_xmin`, `HR_xmax`, `O2Sat_xmin`, `O2Sat_xmax`, `Temp_xmin`, `Temp_xmax`, `SBP_xmin`, `SBP_xmax`, `MAP_xmin`, `MAP_xmax`, `DBP_xmin`, `DBP_xmax`, `Resp_xmin`, `Resp_xmax`, `EtCO2_xmin`, `EtCO2_xmax`, `BaseExcess_xmin`, `BaseExcess_xmax`, `HCO3_xmin`, `HCO3_xmax`, `FiO2_xmin`, `FiO2_xmax`, `pH_xmin`, `pH_xmax`, `PaCO2_xmin`, `PaCO2_xmax`, `SaO2_xmin`, `SaO2_xmax`, `AST_xmin`, `AST_xmax`, `BUN_xmin`, `BUN_xmax`, `Alkalinephos_xmin`, `Alkalinephos_xmax`, `Calcium_xmin`, `Calcium_xmax`, `Chloride_xmin`, `Chloride_xmax`, `Creatinine_xmin`, `Creatinine_xmax`, `Bilirubin_direct_xmin`, `Bilirubin_direct_xmax`, `Glucose_xmin`, `Glucose_xmax`, `Lactate_xmin`, `Lactate_xmax`, `Magnesium_xmin`, `Magnesium_xmax`, `Phosphate_xmin`, `Phosphate_xmax`, `Potassium_xmin`, `Potassium_xmax`, `Bilirubin_total_xmin`, `Bilirubin_total_xmax`, `TroponinI_xmin`, `TroponinI_xmax`, `Hct_xmin`, `Hct_xmax`, `Hgb_xmin`, `Hgb_xmax`, `PTT_xmin`, `PTT_xmax`, `WBC_xmin`, `WBC_xmax`, `Fibrinogen_xmin`, `Fibrinogen_xmax`, `Platelets_xmin`, `Platelets_xmax`, `Lactate_d12`, `Creatinine_d12`, `Platelets_d12`, `WBC_d12`, `Bilirubin_total_d12`, `BUN_d12`, `pH_d12`, `HR_mean3`, `HR_mean12`, `HR_slope6`, `O2Sat_mean3`, `O2Sat_mean12`, `O2Sat_slope6`, `Temp_mean3`, `Temp_mean12`, `Temp_slope6`, `SBP_mean3`, `SBP_mean12`, `SBP_slope6`, `MAP_mean3`, `MAP_mean12`, `MAP_slope6`, `DBP_mean3`, `DBP_mean12`, `DBP_slope6`, `Resp_mean3`, `Resp_mean12`, `Resp_slope6`, `EtCO2_mean3`, `EtCO2_mean12`, `EtCO2_slope6`, `Lactate_dobs`, `Creatinine_dobs`, `Platelets_dobs`, `WBC_dobs`, `Bilirubin_total_dobs`, `BUN_dobs`, `pH_dobs`, `HCO3_dobs`, `Glucose_dobs`, `Hgb_dobs`, `Potassium_dobs`, `Temp_dobs`

### `composite` (35)

`shock_index`, `pulse_pressure`, `sirs`, `qsofa`, `map_low`, `spo2_low`, `organ_dys_count`, `sirs_hours_so_far`, `sirs_mean6`, `sirs_max6`, `sirs_max24`, `sirs_cummax`, `qsofa_max6`, `qsofa_max24`, `qsofa_cummax`, `organ_dys_count_max6`, `organ_dys_count_max24`, `organ_dys_count_cummax`, `hr_hi_n6`, `hr_hi_n24`, `rr_hi_n6`, `rr_hi_n24`, `temp_ab_n6`, `temp_ab_n24`, `map_low_n6`, `map_low_n24`, `sbp_low_n6`, `sbp_low_n24`, `spo2_low_n6`, `spo2_low_n24`, `t_since_first_map_low`, `t_since_first_lactate_hi`, `t_since_first_sirs2`, `bun_creat`, `sao2_fio2`

### `static` (2)

`Age`, `Gender`

### `care` (4)

`ICULOS`, `HospAdmTime`, `Unit1`, `Unit2`
