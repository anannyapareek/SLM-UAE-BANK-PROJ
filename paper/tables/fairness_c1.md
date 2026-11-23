# Fairness, Comparison 1 (test, session level)

| Attribute / group | Sessions | Positives | FPR XGBoost (C1) | FPR SLM (C1) | Recall XGBoost (C1) | Recall SLM (C1) |
|---|---|---|---|---|---|---|
| all / all | 95,570 | 243 | 0.0559 ± 0.0017 | 0.0546 ± 0.0005 | 0.639 ± 0.006 | 0.418 ± 0.101 |
| role / ComputerScientist | 2,708 | 26 | 0.0092 ± 0.0008 | 0.0111 ± 0.0192 | 0.833 ± 0.022 | 0.205 ± 0.355 |
| role / HardwareEngineer | 675 | 29 | 0.0093 ± 0.0056 | 0.0330 ± 0.0572 | 0.035 ± 0.035 | 0.161 ± 0.279 |
| role / ITAdmin | 9,651 | 5 | 0.1330 ± 0.0142 | 0.0923 ± 0.0055 | 1.000 ± 0.000 | 0.400 ± 0.000 |
| role / ProductionLineWorker | 15,966 | 34 | 0.0684 ± 0.0022 | 0.0603 ± 0.0018 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| role / Salesman | 13,307 | 33 | 0.0480 ± 0.0026 | 0.0508 ± 0.0157 | 0.434 ± 0.046 | 0.778 ± 0.035 |
| role / SoftwareEngineer | 3,717 | 24 | 0.0029 ± 0.0018 | 0.0000 ± 0.0000 | 0.569 ± 0.064 | 0.000 ± 0.000 |
| role / SystemsEngineer | 1,776 | 28 | 0.0008 ± 0.0009 | 0.0019 ± 0.0033 | 0.286 ± 0.062 | 0.143 ± 0.247 |
| role / Technician | 13,619 | 30 | 0.0736 ± 0.0010 | 0.0691 ± 0.0018 | 1.000 ± 0.000 | 0.933 ± 0.033 |
| role / TestEngineer | 1,644 | 29 | 0.0275 ± 0.0029 | 0.0023 ± 0.0029 | 0.943 ± 0.020 | 0.035 ± 0.060 |
| department / 2 - Sales | 14,084 | 33 | 0.0477 ± 0.0025 | 0.0511 ± 0.0136 | 0.434 ± 0.046 | 0.778 ± 0.035 |
| department / 2 - SoftwareManagement | 8,868 | 50 | 0.0442 ± 0.0002 | 0.0407 ± 0.0034 | 0.707 ± 0.042 | 0.107 ± 0.185 |
| department / 3 - Assembly | 16,785 | 34 | 0.0692 ± 0.0019 | 0.0626 ± 0.0018 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| department / 3 - Engineering | 11,706 | 86 | 0.0218 ± 0.0021 | 0.0351 ± 0.0106 | 0.422 ± 0.027 | 0.112 ± 0.195 |
| department / 3 - FieldService | 13,904 | 30 | 0.0692 ± 0.0010 | 0.0661 ± 0.0024 | 1.000 ± 0.000 | 0.933 ± 0.033 |
| department / 6 - Security | 11,728 | 9 | 0.1150 ± 0.0115 | 0.0786 ± 0.0047 | 0.704 ± 0.064 | 0.222 ± 0.000 |
| functional_unit / 1 - Adminstration | 15,237 | 9 | 0.0886 ± 0.0088 | 0.0629 ± 0.0053 | 0.704 ± 0.064 | 0.222 ± 0.000 |
| functional_unit / 2 - ResearchAndEngineering | 29,531 | 136 | 0.0341 ± 0.0014 | 0.0417 ± 0.0050 | 0.527 ± 0.026 | 0.110 ± 0.191 |
| functional_unit / 3 - Manufacturing | 19,075 | 34 | 0.0706 ± 0.0018 | 0.0663 ± 0.0014 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| functional_unit / 5 - SalesAndMarketing | 28,459 | 64 | 0.0574 ± 0.0017 | 0.0577 ± 0.0058 | 0.693 ± 0.024 | 0.839 ± 0.024 |

Test split, session level; alert at each seed's validation 5%-FPR threshold; mean ± std over 3 seeds. Groups with at least 5 positive sessions (recall defined), as in the fairness reports; the full lists are in results/comparison*/fairness_*_test.csv.
