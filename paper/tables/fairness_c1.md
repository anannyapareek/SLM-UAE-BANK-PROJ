# Fairness, Comparison 1 (test, session level)

| Attribute / group | Sessions | Positives | FPR XGBoost (C1) | FPR SLM (C1) | Recall XGBoost (C1) | Recall SLM (C1) |
|---|---|---|---|---|---|---|
| all / all | 95,570 | 243 | 0.0559 ± 0.0017 | 0.0546 ± 0.0005 | 0.639 ± 0.006 | 0.418 ± 0.101 |
| role / Accountant | 292 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / AdministrativeAssistant | 3,207 | 1 | 0.0001 ± 0.0002 | 0.0045 ± 0.0042 | – | – |
| role / AssemblySupervisor | 596 | 0 | 0.0028 ± 0.0010 | 0.0280 ± 0.0010 | – | – |
| role / Attorney | 330 | 0 | 0.0010 ± 0.0017 | 0.0697 ± 0.0259 | – | – |
| role / ChiefEngineer | 727 | 0 | 0.0005 ± 0.0008 | 0.0000 ± 0.0000 | – | – |
| role / ComputerProgrammer | 2,136 | 0 | 0.1261 ± 0.0019 | 0.1573 ± 0.0089 | – | – |
| role / ComputerScientist | 2,708 | 26 | 0.0092 ± 0.0008 | 0.0111 ± 0.0192 | 0.833 ± 0.022 | 0.205 ± 0.355 |
| role / ComputerTrainer | 230 | 0 | 0.0000 ± 0.0000 | 0.0594 ± 0.0955 | – | – |
| role / Director | 1,262 | 0 | 0.0013 ± 0.0005 | 0.0127 ± 0.0114 | – | – |
| role / ElectricalEngineer | 2,590 | 0 | 0.0936 ± 0.0016 | 0.1254 ± 0.0076 | – | – |
| role / Engineer | 213 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / FieldServiceEngineer | 669 | 0 | 0.0015 ± 0.0015 | 0.0309 ± 0.0199 | – | – |
| role / HardwareEngineer | 675 | 29 | 0.0093 ± 0.0056 | 0.0330 ± 0.0572 | 0.035 ± 0.035 | 0.161 ± 0.279 |
| role / HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / HumanResourceSpecialist | 1,192 | 0 | 0.0000 ± 0.0000 | 0.0006 ± 0.0010 | – | – |
| role / ITAdmin | 9,651 | 5 | 0.1330 ± 0.0142 | 0.0923 ± 0.0055 | 1.000 ± 0.000 | 0.400 ± 0.000 |
| role / IndustrialEngineer | 596 | 0 | 0.1488 ± 0.0079 | 0.1980 ± 0.0501 | – | – |
| role / LabManager | 570 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / Manager | 2,491 | 2 | 0.0992 ± 0.0022 | 0.0524 ± 0.0103 | – | – |
| role / MaterialsEngineer | 582 | 0 | 0.0006 ± 0.0010 | 0.0172 ± 0.0241 | – | – |
| role / Mathematician | 2,462 | 0 | 0.1118 ± 0.0022 | 0.1010 ± 0.0121 | – | – |
| role / MechanicalEngineer | 2,453 | 0 | 0.0035 ± 0.0012 | 0.0276 ± 0.0264 | – | – |
| role / NursePractitioner | 305 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| role / Physicist | 1,784 | 0 | 0.0062 ± 0.0015 | 0.0817 ± 0.0122 | – | – |
| role / ProductionLineWorker | 15,966 | 34 | 0.0684 ± 0.0022 | 0.0603 ± 0.0018 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| role / PurchasingClerk | 481 | 0 | 0.0007 ± 0.0012 | 0.0922 ± 0.0428 | – | – |
| role / Salesman | 13,307 | 33 | 0.0480 ± 0.0026 | 0.0508 ± 0.0157 | 0.434 ± 0.046 | 0.778 ± 0.035 |
| role / Scientist | 3,761 | 0 | 0.0195 ± 0.0047 | 0.0176 ± 0.0105 | – | – |
| role / SecurityGuard | 1,558 | 2 | 0.0062 ± 0.0021 | 0.0163 ± 0.0127 | – | – |
| role / SoftwareEngineer | 3,717 | 24 | 0.0029 ± 0.0018 | 0.0000 ± 0.0000 | 0.569 ± 0.064 | 0.000 ± 0.000 |
| role / Statistician | 372 | 0 | 0.0000 ± 0.0000 | 0.0896 ± 0.0056 | – | – |
| role / SystemsEngineer | 1,776 | 28 | 0.0008 ± 0.0009 | 0.0019 ± 0.0033 | 0.286 ± 0.062 | 0.143 ± 0.247 |
| role / Technician | 13,619 | 30 | 0.0736 ± 0.0010 | 0.0691 ± 0.0018 | 1.000 ± 0.000 | 0.933 ± 0.033 |
| role / TestEngineer | 1,644 | 29 | 0.0275 ± 0.0029 | 0.0023 ± 0.0029 | 0.943 ± 0.020 | 0.035 ± 0.060 |
| role / VicePresident | 627 | 0 | 0.0011 ± 0.0018 | 0.0048 ± 0.0058 | – | – |
| department / 1 - Accounting | 380 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 1 - IndustrialEngineering | 702 | 0 | 0.1263 ± 0.0067 | 0.1681 ± 0.0425 | – | – |
| department / 1 - MarketAnalysis | 280 | 1 | 0.0000 ± 0.0000 | 0.0084 ± 0.0075 | – | – |
| department / 1 - Personnel | 992 | 0 | 0.0000 ± 0.0000 | 0.0017 ± 0.0029 | – | – |
| department / 1 - Purchasing | 584 | 0 | 0.0017 ± 0.0000 | 0.0759 ± 0.0353 | – | – |
| department / 1 - Research | 8,767 | 0 | 0.0410 ± 0.0018 | 0.0525 ± 0.0021 | – | – |
| department / 2 - EmployeeRelations | 595 | 0 | 0.0000 ± 0.0000 | 0.0095 ± 0.0093 | – | – |
| department / 2 - ManufacturingEngineering | 952 | 0 | 0.1001 ± 0.0034 | 0.1019 ± 0.0000 | – | – |
| department / 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 2 - Pricing | 381 | 0 | 0.0026 ± 0.0045 | 0.0875 ± 0.0055 | – | – |
| department / 2 - Sales | 14,084 | 33 | 0.0477 ± 0.0025 | 0.0511 ± 0.0136 | 0.434 ± 0.046 | 0.778 ± 0.035 |
| department / 2 - SoftwareManagement | 8,868 | 50 | 0.0442 ± 0.0002 | 0.0407 ± 0.0034 | 0.707 ± 0.042 | 0.107 ± 0.185 |
| department / 3 - Assembly | 16,785 | 34 | 0.0692 ± 0.0019 | 0.0626 ± 0.0018 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| department / 3 - Contracts | 540 | 0 | 0.0006 ± 0.0011 | 0.0494 ± 0.0204 | – | – |
| department / 3 - Engineering | 11,706 | 86 | 0.0218 ± 0.0021 | 0.0351 ± 0.0106 | 0.422 ± 0.027 | 0.112 ± 0.195 |
| department / 3 - FieldService | 13,904 | 30 | 0.0692 ± 0.0010 | 0.0661 ± 0.0024 | 1.000 ± 0.000 | 0.933 ± 0.033 |
| department / 3 - FinancialPlanning | 489 | 0 | 0.0041 ± 0.0020 | 0.0368 ± 0.0106 | – | – |
| department / 3 - Training | 805 | 0 | 0.0000 ± 0.0000 | 0.0327 ± 0.0253 | – | – |
| department / 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 5 - Medical | 495 | 0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | – | – |
| department / 6 - Security | 11,728 | 9 | 0.1150 ± 0.0115 | 0.0786 ± 0.0047 | 0.704 ± 0.064 | 0.222 ± 0.000 |
| department / unknown | 1,387 | 0 | 0.0005 ± 0.0008 | 0.0022 ± 0.0026 | – | – |
| functional_unit / 1 - Adminstration | 15,237 | 9 | 0.0886 ± 0.0088 | 0.0629 ± 0.0053 | 0.704 ± 0.064 | 0.222 ± 0.000 |
| functional_unit / 2 - ResearchAndEngineering | 29,531 | 136 | 0.0341 ± 0.0014 | 0.0417 ± 0.0050 | 0.527 ± 0.026 | 0.110 ± 0.191 |
| functional_unit / 3 - Manufacturing | 19,075 | 34 | 0.0706 ± 0.0018 | 0.0663 ± 0.0014 | 0.971 ± 0.000 | 0.912 ± 0.051 |
| functional_unit / 4 - Finance | 1,357 | 0 | 0.0015 ± 0.0007 | 0.0133 ± 0.0038 | – | – |
| functional_unit / 5 - SalesAndMarketing | 28,459 | 64 | 0.0574 ± 0.0017 | 0.0577 ± 0.0058 | 0.693 ± 0.024 | 0.839 ± 0.024 |
| functional_unit / 6 - PurchasingAndContracts | 1,721 | 0 | 0.0014 ± 0.0009 | 0.0606 ± 0.0092 | – | – |

Test split, session level; alert at each seed's validation 5%-FPR threshold; mean ± std over 3 seeds. FPR for groups with ≥ 200 sessions; recall for groups with ≥ 5 positive sessions. – = fewer positive sessions. Runbook size rule; full lists in results/comparison*/fairness_*_test.csv.
