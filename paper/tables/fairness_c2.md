# Fairness, Comparison 2 (test, session level)

| Attribute / group | Sessions | Positives | FPR XGBoost text (C2) | FPR SLM text (C2) | Recall XGBoost text (C2) | Recall SLM text (C2) |
|---|---|---|---|---|---|---|
| all / all | 95,570 | 243 | 0.0600 ± 0.0017 | 0.0589 ± 0.0004 | 0.722 ± 0.026 | 0.840 ± 0.022 |
| role / ComputerScientist | 2,708 | 26 | 0.0055 ± 0.0021 | 0.0454 ± 0.0018 | 0.500 ± 0.038 | 0.885 ± 0.000 |
| role / HardwareEngineer | 675 | 29 | 0.0330 ± 0.0103 | 0.1889 ± 0.0067 | 0.264 ± 0.080 | 0.586 ± 0.000 |
| role / ITAdmin | 9,651 | 5 | 0.2431 ± 0.0106 | 0.0915 ± 0.0063 | 1.000 ± 0.000 | 0.800 ± 0.000 |
| role / ProductionLineWorker | 15,966 | 34 | 0.0648 ± 0.0036 | 0.0489 ± 0.0037 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| role / Salesman | 13,307 | 33 | 0.0255 ± 0.0035 | 0.0408 ± 0.0017 | 0.697 ± 0.080 | 0.869 ± 0.063 |
| role / SoftwareEngineer | 3,717 | 24 | 0.0009 ± 0.0007 | 0.0332 ± 0.0004 | 0.722 ± 0.251 | 0.958 ± 0.000 |
| role / SystemsEngineer | 1,776 | 28 | 0.0114 ± 0.0015 | 0.0343 ± 0.0060 | 0.679 ± 0.036 | 0.821 ± 0.036 |
| role / Technician | 13,619 | 30 | 0.0510 ± 0.0026 | 0.0534 ± 0.0063 | 1.000 ± 0.000 | 0.967 ± 0.033 |
| role / TestEngineer | 1,644 | 29 | 0.0194 ± 0.0068 | 0.0809 ± 0.0092 | 0.839 ± 0.053 | 0.667 ± 0.072 |
| department / 2 - Sales | 14,084 | 33 | 0.0260 ± 0.0036 | 0.0401 ± 0.0025 | 0.697 ± 0.080 | 0.869 ± 0.063 |
| department / 2 - SoftwareManagement | 8,868 | 50 | 0.0276 ± 0.0029 | 0.0683 ± 0.0021 | 0.607 ± 0.141 | 0.920 ± 0.000 |
| department / 3 - Assembly | 16,785 | 34 | 0.0650 ± 0.0041 | 0.0504 ± 0.0037 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| department / 3 - Engineering | 11,706 | 86 | 0.0337 ± 0.0015 | 0.0639 ± 0.0018 | 0.593 ± 0.012 | 0.690 ± 0.034 |
| department / 3 - FieldService | 13,904 | 30 | 0.0447 ± 0.0027 | 0.0498 ± 0.0057 | 1.000 ± 0.000 | 0.967 ± 0.033 |
| department / 6 - Security | 11,728 | 9 | 0.2074 ± 0.0083 | 0.0877 ± 0.0063 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| functional_unit / 1 - Adminstration | 15,237 | 9 | 0.1611 ± 0.0065 | 0.0793 ± 0.0063 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| functional_unit / 2 - ResearchAndEngineering | 29,531 | 136 | 0.0345 ± 0.0016 | 0.0707 ± 0.0006 | 0.598 ± 0.048 | 0.774 ± 0.021 |
| functional_unit / 3 - Manufacturing | 19,075 | 34 | 0.0649 ± 0.0039 | 0.0495 ± 0.0031 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| functional_unit / 5 - SalesAndMarketing | 28,459 | 64 | 0.0348 ± 0.0025 | 0.0443 ± 0.0030 | 0.844 ± 0.041 | 0.917 ± 0.039 |

Test split, session level; alert at each seed's validation 5%-FPR threshold; mean ± std over 3 seeds. Groups with at least 5 positive sessions (recall defined), as in the fairness reports; the full lists are in results/comparison*/fairness_*_test.csv.
