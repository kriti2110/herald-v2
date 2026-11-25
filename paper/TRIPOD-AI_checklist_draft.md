# TRIPOD+AI checklist: DRAFT for HERALD-v2

**Status: draft prepared from the manuscript text. It must be checked by the authors against the official TRIPOD+AI form
(Collins et al., BMJ 2024;385:e078378, https://www.tripod-statement.org) before submission.** Topics below follow the structure of the
official checklist; copy the entries into the official template and use its exact item numbers. Locations refer to the manuscript sections.

"Reported" = stated in the manuscript. "Gap" = not reported, or reported only partly; see the note.

## Title and abstract
| Topic | Status | Where / note |
|---|---|---|
| Title identifies a prediction-model study, population, outcome | Reported | Title: early prediction and detection of sepsis, PhysioNet/CinC 2019 |
| Structured abstract | Partly | Single-paragraph abstract. Elsevier journals may want the TRIPOD+AI-for-Abstracts items (data source, setting, outcome, n, performance with CIs): all numbers are present, headings are not |

## Introduction
| Topic | Status | Where / note |
|---|---|---|
| Healthcare context and rationale, existing models | Reported | Introduction; Related Work, Table 1 |
| Target population and intended use / users | Partly | Population: adult ICU stays (challenge data). Intended use is research only ("research prototypes, not clinical decision tools", Discussion, Ethics). No intended-user statement beyond that |
| Known health inequalities between groups | **Gap** | Not discussed. Consider one or two sentences in the Introduction or Discussion |
| Objectives, development and/or validation | Reported | Introduction (four contributions); Sec. 3.2 |

## Methods
| Topic | Status | Where / note |
|---|---|---|
| Data sources, rationale, representativeness | Reported | Sec. 3.1 (PhysioNet/CinC 2019 sets A and B, two hospital systems). Representativeness limits: Limitations (1) |
| Dates of data collection | **Gap** | The public challenge data do not give calendar dates; state this explicitly |
| Study setting | Reported | Sec. 3.1 (ICU stays, two hospital systems) |
| Eligibility criteria | Partly | All challenge training stays, after removing byte-identical duplicates (Sec. 3.1). No other exclusions stated |
| Treatments received and how handled | **Gap** | Not available in the data (antibiotic and culture timestamps are unavailable, Limitations (2)); say that treatments were not modelled |
| Data preparation, cleaning, quality checks | Reported | Sec. 3.1 (duplicate removal), 3.3 (forward-fill, causal windows) |
| Outcome definition, time horizon, how assessed | Reported | Sec. 3.2 (SepsisLabel; 6 h early prediction, hourly detection, stay-level). Label is the challenge's, not clinically re-verified: Limitations (2) |
| Predictors: definition, timing, measurement | Reported | Sec. 3.3 and Table of features (370 causal features); prefix-validity definition, Eq. 1 |
| Sample size and its justification | **Gap** | All available stays were used; no formal sample-size calculation. State this |
| Missing data handling | Partly | Forward-fill (LOCF) and measurement-timing features (Sec. 3.3); the remaining missing values are passed to LightGBM unimputed (verified in `hv2/features.py` and the training scripts: the only `fillna(0)` is inside a running-sum computation). Add one sentence saying so explicitly |
| Analytical methods: model type, hyper-parameters, selection | Reported | Sec. 3.4 to 3.6, hyper-parameter table; tuning on validation patients only (Sec. 3.7) |
| Class imbalance | Reported | Natural prevalence kept; positive-class weighting tested and rejected (negative results) |
| Handling of predictors / data leakage | Reported | Sec. 3.3, 3.8 (prefix-invariance audit with negative control) |
| Model output and how risk is produced | Reported | Two-stage score; isotonic calibration (Sec. 3.7); thresholds illustrative (Sec. 3.9) |
| Training versus evaluation data | Reported | Patient-level 5-fold CV; hospital A to B and B to A transfer (Sec. 3.7, 4.3) |
| Performance measures | Reported | AUROC, AUPRC, calibration (ECE, Brier skill, slope), decision curves, alert burden (Sec. 3.7, 3.9) |
| Uncertainty | Reported | Patient-clustered bootstrap, 1000 replicates (300 for subgroups) |
| Fairness / subgroup analysis | Partly | Age band, sex, hospital (Sec. 3.9, 4.7). Race and ethnicity not in the data (stated) |
| Model updating / recalibration | Partly | Cross-hospital calibration reported; recalibration recommended before use at a new site (Sec. 4.6) but not performed |
| Ethical approval and consent | Reported | Public de-identified PhysioNet data (Discussion, Ethics). Add "no additional ethical approval was required" if that is what your institution confirms |
| Patient and public involvement | **Gap** | None. State "No patient or public involvement" |

## Results
| Topic | Status | Where / note |
|---|---|---|
| Participant flow, numbers, outcome counts | Reported | Sec. 3.1, protocol table (40,336 stays, 2,932 septic, row-level positives) |
| Participant characteristics | Partly | Hospital split and prevalence reported; age and sex distributions appear only as subgroup sizes (subgroup table). A descriptive table of age/sex would close this |
| Model performance with CIs | Reported | Sec. 4.1 to 4.8, tables with intervals |
| Subgroup and cross-hospital performance | Reported | Sec. 4.3, 4.7 |
| Calibration | Reported | Sec. 4.6 (cross-fold isotonic), calibration across hospitals |
| Model specification / ability to reproduce | Reported | Methods, hyper-parameter table, repository |

## Discussion
| Topic | Status | Where / note |
|---|---|---|
| Limitations (bias, generalisability, label quality) | Reported | Limitations section (8 points) |
| Interpretation, clinical use, next steps | Reported | Discussion; Future work |

## Open science
| Topic | Status | Where / note |
|---|---|---|
| Funding | **Gap** | Placeholder in manuscript, authors to complete |
| Conflicts of interest | **Gap** | Placeholder in manuscript, authors to complete |
| Protocol | Reported (none) | No prior protocol; the Process Log appendix records decisions in order |
| Registration | Reported (none) | Not registered; state "not applicable" |
| Data availability | Reported | PhysioNet/CinC 2019 (public); fold assignments and saved predictions in the repository |
| Code availability | Reported | https://github.com/kriti2110/herald-v2 (private until the authors make it public) |

## Items for the authors to close before submission
1. Declarations: funding, competing interests, CRediT.
2. Add: no calendar dates available; no sample-size calculation; no patient or public involvement; not registered.
3. Add one or two sentences on health inequalities and intended users.
4. Add an explicit missing-data sentence to Methods: after forward-fill, remaining missing values are left to LightGBM's native missing-value handling (checked against the code).
5. Optionally add a descriptive table of age and sex for the cohort.
6. Transfer these entries into the official TRIPOD+AI template with its exact item numbers.
