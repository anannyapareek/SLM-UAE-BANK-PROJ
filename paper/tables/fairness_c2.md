# Fairness, Comparison 2 (test, session level)

| Attribute / group | Sessions | Positives | FPR XGBoost text (C2) | FPR SLM text (C2) | Recall XGBoost text (C2) | Recall SLM text (C2) |
|---|---|---|---|---|---|---|
| all / all | 95,570 | 243 | 0.0600 ± 0.0017 | 0.0589 ± 0.0004 | 0.722 ± 0.026 | 0.840 ± 0.022 |
| role / Accountant | 292 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / AdministrativeAssistant | 3,207 | 1 | 0.0020 ± 0.0005 | 0.0183 ± 0.0067 | – | – |
| role / AssemblySupervisor | 596 | 0 | 0.0000 ± 0.0000 | 0.0291 ± 0.0042 | – | – |
| role / Attorney | 330 | 0 | 0.0313 ± 0.0070 | 0.0091 ± 0.0000 | – | – |
| role / ChiefEngineer | 727 | 0 | 0.0083 ± 0.0060 | 0.0830 ± 0.0147 | – | – |
| role / ComputerProgrammer | 2,136 | 0 | 0.0757 ± 0.0143 | 0.0999 ± 0.0027 | – | – |
| role / ComputerScientist | 2,708 | 26 | 0.0055 ± 0.0021 | 0.0454 ± 0.0018 | 0.500 ± 0.038 | 0.885 ± 0.000 |
| role / ComputerTrainer | 230 | 0 | 0.0072 ± 0.0066 | 0.1754 ± 0.0100 | – | – |
| role / Director | 1,262 | 0 | 0.0005 ± 0.0005 | 0.0000 ± 0.0000 | – | – |
| role / ElectricalEngineer | 2,590 | 0 | 0.1106 ± 0.0040 | 0.0356 ± 0.0054 | – | – |
| role / Engineer | 213 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / FieldServiceEngineer | 669 | 0 | 0.0055 ± 0.0023 | 0.0538 ± 0.0068 | – | – |
| role / HardwareEngineer | 675 | 29 | 0.0330 ± 0.0103 | 0.1889 ± 0.0067 | 0.264 ± 0.080 | 0.586 ± 0.000 |
| role / HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / HumanResourceSpecialist | 1,192 | 0 | 0.0025 ± 0.0008 | 0.0333 ± 0.0049 | – | – |
| role / ITAdmin | 9,651 | 5 | 0.2431 ± 0.0106 | 0.0915 ± 0.0063 | 1.000 ± 0.000 | 0.800 ± 0.000 |
| role / IndustrialEngineer | 596 | 0 | 0.1091 ± 0.0191 | 0.1186 ± 0.0137 | – | – |
| role / LabManager | 570 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / Manager | 2,491 | 2 | 0.0832 ± 0.0011 | 0.0940 ± 0.0123 | – | – |
| role / MaterialsEngineer | 582 | 0 | 0.0023 ± 0.0040 | 0.0624 ± 0.0114 | – | – |
| role / Mathematician | 2,462 | 0 | 0.1060 ± 0.0134 | 0.0988 ± 0.0038 | – | – |
| role / MechanicalEngineer | 2,453 | 0 | 0.0117 ± 0.0017 | 0.0737 ± 0.0015 | – | – |
| role / NursePractitioner | 305 | 0 | 0.0011 ± 0.0019 | 0.0000 ± 0.0000 | – | – |
| role / Physicist | 1,784 | 0 | 0.0125 ± 0.0044 | 0.1362 ± 0.0037 | – | – |
| role / ProductionLineWorker | 15,966 | 34 | 0.0648 ± 0.0036 | 0.0489 ± 0.0037 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| role / PurchasingClerk | 481 | 0 | 0.0014 ± 0.0012 | 0.1275 ± 0.0295 | – | – |
| role / Salesman | 13,307 | 33 | 0.0255 ± 0.0035 | 0.0408 ± 0.0017 | 0.697 ± 0.080 | 0.869 ± 0.063 |
| role / Scientist | 3,761 | 0 | 0.0256 ± 0.0078 | 0.0653 ± 0.0027 | – | – |
| role / SecurityGuard | 1,558 | 2 | 0.0216 ± 0.0035 | 0.0718 ± 0.0077 | – | – |
| role / SoftwareEngineer | 3,717 | 24 | 0.0009 ± 0.0007 | 0.0332 ± 0.0004 | 0.722 ± 0.251 | 0.958 ± 0.000 |
| role / Statistician | 372 | 0 | 0.0197 ± 0.0138 | 0.1613 ± 0.0027 | – | – |
| role / SystemsEngineer | 1,776 | 28 | 0.0114 ± 0.0015 | 0.0343 ± 0.0060 | 0.679 ± 0.036 | 0.821 ± 0.036 |
| role / Technician | 13,619 | 30 | 0.0510 ± 0.0026 | 0.0534 ± 0.0063 | 1.000 ± 0.000 | 0.967 ± 0.033 |
| role / TestEngineer | 1,644 | 29 | 0.0194 ± 0.0068 | 0.0809 ± 0.0092 | 0.839 ± 0.053 | 0.667 ± 0.072 |
| role / VicePresident | 627 | 0 | 0.0165 ± 0.0056 | 0.0579 ± 0.0120 | – | – |
| department / 1 - Accounting | 380 | 0 | 0.0009 ± 0.0015 | 0.0000 ± 0.0000 | – | – |
| department / 1 - IndustrialEngineering | 702 | 0 | 0.0926 ± 0.0162 | 0.1007 ± 0.0116 | – | – |
| department / 1 - MarketAnalysis | 280 | 1 | 0.0072 ± 0.0072 | 0.0119 ± 0.0041 | – | – |
| department / 1 - Personnel | 992 | 0 | 0.0030 ± 0.0010 | 0.0528 ± 0.0150 | – | – |
| department / 1 - Purchasing | 584 | 0 | 0.0023 ± 0.0010 | 0.1050 ± 0.0243 | – | – |
| department / 1 - Research | 8,767 | 0 | 0.0433 ± 0.0074 | 0.0835 ± 0.0013 | – | – |
| department / 2 - EmployeeRelations | 595 | 0 | 0.0006 ± 0.0010 | 0.0151 ± 0.0133 | – | – |
| department / 2 - ManufacturingEngineering | 952 | 0 | 0.0875 ± 0.0082 | 0.0287 ± 0.0112 | – | – |
| department / 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 2 - Pricing | 381 | 0 | 0.0219 ± 0.0084 | 0.1575 ± 0.0026 | – | – |
| department / 2 - Sales | 14,084 | 33 | 0.0260 ± 0.0036 | 0.0401 ± 0.0025 | 0.697 ± 0.080 | 0.869 ± 0.063 |
| department / 2 - SoftwareManagement | 8,868 | 50 | 0.0276 ± 0.0029 | 0.0683 ± 0.0021 | 0.607 ± 0.141 | 0.920 ± 0.000 |
| department / 3 - Assembly | 16,785 | 34 | 0.0650 ± 0.0041 | 0.0504 ± 0.0037 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| department / 3 - Contracts | 540 | 0 | 0.0278 ± 0.0111 | 0.0056 ± 0.0000 | – | – |
| department / 3 - Engineering | 11,706 | 86 | 0.0337 ± 0.0015 | 0.0639 ± 0.0018 | 0.593 ± 0.012 | 0.690 ± 0.034 |
| department / 3 - FieldService | 13,904 | 30 | 0.0447 ± 0.0027 | 0.0498 ± 0.0057 | 1.000 ± 0.000 | 0.967 ± 0.033 |
| department / 3 - FinancialPlanning | 489 | 0 | 0.0136 ± 0.0031 | 0.0136 ± 0.0077 | – | – |
| department / 3 - Training | 805 | 0 | 0.0112 ± 0.0022 | 0.1023 ± 0.0026 | – | – |
| department / 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 5 - Medical | 495 | 0 | 0.0007 ± 0.0012 | 0.0000 ± 0.0000 | – | – |
| department / 6 - Security | 11,728 | 9 | 0.2074 ± 0.0083 | 0.0877 ± 0.0063 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| department / unknown | 1,387 | 0 | 0.0075 ± 0.0025 | 0.0262 ± 0.0054 | – | – |
| functional_unit / 1 - Adminstration | 15,237 | 9 | 0.1611 ± 0.0065 | 0.0793 ± 0.0063 | 0.667 ± 0.000 | 0.667 ± 0.000 |
| functional_unit / 2 - ResearchAndEngineering | 29,531 | 136 | 0.0345 ± 0.0016 | 0.0707 ± 0.0006 | 0.598 ± 0.048 | 0.774 ± 0.021 |
| functional_unit / 3 - Manufacturing | 19,075 | 34 | 0.0649 ± 0.0039 | 0.0495 ± 0.0031 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| functional_unit / 4 - Finance | 1,357 | 0 | 0.0052 ± 0.0007 | 0.0049 ± 0.0028 | – | – |
| functional_unit / 5 - SalesAndMarketing | 28,459 | 64 | 0.0348 ± 0.0025 | 0.0443 ± 0.0030 | 0.844 ± 0.041 | 0.917 ± 0.039 |
| functional_unit / 6 - PurchasingAndContracts | 1,721 | 0 | 0.0143 ± 0.0048 | 0.0722 ± 0.0085 | – | – |

Test split, session level; alert at each seed's validation 5%-FPR threshold; mean ± std over 3 seeds. FPR for groups with ≥ 200 sessions; recall for groups with ≥ 5 positive sessions. – = fewer positive sessions. Runbook size rule; full lists in results/comparison*/fairness_*_test.csv.
