# Fairness on the test split (runbook 6.4)

Session units, max aggregation; alert at each seed's validation 5%-FPR threshold. Mean ± std over 3 seeds. Groups: the pre-declared fairness proxies role, department and functional_unit (tenure was dropped on 2026-11-04 and is not available). Shown only for groups with >= 200 sessions; recall only where a group has >= 5 positive sessions. Flags: the lowest and highest FPR (and recall) in each attribute; **small sample** marks such an extreme from a group with fewer than 500 sessions.

## SLM (Qwen2.5-1.5B LoRA)

### all

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| all | 95,570 | 243 | 0.0546 ± 0.0005 | 0.418 ± 0.101 |  |

### role

FPR range: lowest SoftwareEngineer 0.000 (n = 3,717); ChiefEngineer 0.000 (n = 727); LabManager 0.000 (n = 570); HealthSafetyEngineer 0.000 (n = 322, small sample); NursePractitioner 0.000 (n = 305, small sample); Accountant 0.000 (n = 292, small sample); Engineer 0.000 (n = 213, small sample); highest IndustrialEngineer 0.198 (n = 596).
Recall range: lowest SoftwareEngineer 0.000 (n = 3,717); highest Technician 0.933 (n = 13,619).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| ProductionLineWorker | 15,966 | 34 | 0.0603 ± 0.0018 | 0.912 ± 0.051 |  |
| Technician | 13,619 | 30 | 0.0691 ± 0.0018 | 0.933 ± 0.033 | highest recall |
| Salesman | 13,307 | 33 | 0.0508 ± 0.0157 | 0.778 ± 0.035 |  |
| ITAdmin | 9,651 | 5 | 0.0923 ± 0.0055 | 0.400 ± 0.000 |  |
| Scientist | 3,761 | 0 | 0.0176 ± 0.0105 | n/a (< 5 positives) |  |
| SoftwareEngineer | 3,717 | 24 | 0.0000 ± 0.0000 | 0.000 ± 0.000 | lowest FPR, lowest recall |
| AdministrativeAssistant | 3,207 | 1 | 0.0045 ± 0.0042 | n/a (< 5 positives) |  |
| ComputerScientist | 2,708 | 26 | 0.0111 ± 0.0192 | 0.205 ± 0.355 |  |
| ElectricalEngineer | 2,590 | 0 | 0.1254 ± 0.0076 | n/a (< 5 positives) |  |
| Manager | 2,491 | 2 | 0.0524 ± 0.0103 | n/a (< 5 positives) |  |
| Mathematician | 2,462 | 0 | 0.1010 ± 0.0121 | n/a (< 5 positives) |  |
| MechanicalEngineer | 2,453 | 0 | 0.0276 ± 0.0264 | n/a (< 5 positives) |  |
| ComputerProgrammer | 2,136 | 0 | 0.1573 ± 0.0089 | n/a (< 5 positives) |  |
| Physicist | 1,784 | 0 | 0.0817 ± 0.0122 | n/a (< 5 positives) |  |
| SystemsEngineer | 1,776 | 28 | 0.0019 ± 0.0033 | 0.143 ± 0.247 |  |
| TestEngineer | 1,644 | 29 | 0.0023 ± 0.0029 | 0.035 ± 0.060 |  |
| SecurityGuard | 1,558 | 2 | 0.0163 ± 0.0127 | n/a (< 5 positives) |  |
| Director | 1,262 | 0 | 0.0127 ± 0.0114 | n/a (< 5 positives) |  |
| HumanResourceSpecialist | 1,192 | 0 | 0.0006 ± 0.0010 | n/a (< 5 positives) |  |
| ChiefEngineer | 727 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| HardwareEngineer | 675 | 29 | 0.0330 ± 0.0572 | 0.161 ± 0.279 |  |
| FieldServiceEngineer | 669 | 0 | 0.0309 ± 0.0199 | n/a (< 5 positives) |  |
| VicePresident | 627 | 0 | 0.0048 ± 0.0058 | n/a (< 5 positives) |  |
| AssemblySupervisor | 596 | 0 | 0.0280 ± 0.0010 | n/a (< 5 positives) |  |
| IndustrialEngineer | 596 | 0 | 0.1980 ± 0.0501 | n/a (< 5 positives) | highest FPR |
| MaterialsEngineer | 582 | 0 | 0.0172 ± 0.0241 | n/a (< 5 positives) |  |
| LabManager | 570 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| PurchasingClerk | 481 | 0 | 0.0922 ± 0.0428 | n/a (< 5 positives) |  |
| Statistician | 372 | 0 | 0.0896 ± 0.0056 | n/a (< 5 positives) |  |
| Attorney | 330 | 0 | 0.0697 ± 0.0259 | n/a (< 5 positives) |  |
| HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| NursePractitioner | 305 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| Accountant | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| ComputerTrainer | 230 | 0 | 0.0594 ± 0.0955 | n/a (< 5 positives) |  |
| Engineer | 213 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |

### department

FPR range: lowest 5 - Medical 0.000 (n = 495, small sample); 4 - Safety 0.000 (n = 432, small sample); 4 - QualityControl 0.000 (n = 422, small sample); 1 - Accounting 0.000 (n = 380, small sample); 2 - Payroll 0.000 (n = 292, small sample); highest 1 - IndustrialEngineering 0.168 (n = 702).
Recall range: lowest 2 - SoftwareManagement 0.107 (n = 8,868); highest 3 - FieldService 0.933 (n = 13,904).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 3 - Assembly | 16,785 | 34 | 0.0626 ± 0.0018 | 0.912 ± 0.051 |  |
| 2 - Sales | 14,084 | 33 | 0.0511 ± 0.0136 | 0.778 ± 0.035 |  |
| 3 - FieldService | 13,904 | 30 | 0.0661 ± 0.0024 | 0.933 ± 0.033 | highest recall |
| 6 - Security | 11,728 | 9 | 0.0786 ± 0.0047 | 0.222 ± 0.000 |  |
| 3 - Engineering | 11,706 | 86 | 0.0351 ± 0.0106 | 0.112 ± 0.195 |  |
| 2 - SoftwareManagement | 8,868 | 50 | 0.0407 ± 0.0034 | 0.107 ± 0.185 | lowest recall |
| 1 - Research | 8,767 | 0 | 0.0525 ± 0.0021 | n/a (< 5 positives) |  |
| unknown | 1,387 | 0 | 0.0022 ± 0.0026 | n/a (< 5 positives) |  |
| 1 - Personnel | 992 | 0 | 0.0017 ± 0.0029 | n/a (< 5 positives) |  |
| 2 - ManufacturingEngineering | 952 | 0 | 0.1019 ± 0.0000 | n/a (< 5 positives) |  |
| 3 - Training | 805 | 0 | 0.0327 ± 0.0253 | n/a (< 5 positives) |  |
| 1 - IndustrialEngineering | 702 | 0 | 0.1681 ± 0.0425 | n/a (< 5 positives) | highest FPR |
| 2 - EmployeeRelations | 595 | 0 | 0.0095 ± 0.0093 | n/a (< 5 positives) |  |
| 1 - Purchasing | 584 | 0 | 0.0759 ± 0.0353 | n/a (< 5 positives) |  |
| 3 - Contracts | 540 | 0 | 0.0494 ± 0.0204 | n/a (< 5 positives) |  |
| 5 - Medical | 495 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 3 - FinancialPlanning | 489 | 0 | 0.0368 ± 0.0106 | n/a (< 5 positives) |  |
| 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Pricing | 381 | 0 | 0.0875 ± 0.0055 | n/a (< 5 positives) |  |
| 1 - Accounting | 380 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 1 - MarketAnalysis | 280 | 1 | 0.0084 ± 0.0075 | n/a (< 5 positives) |  |

### functional_unit

FPR range: lowest 4 - Finance 0.013 (n = 1,357); highest 3 - Manufacturing 0.066 (n = 19,075).
Recall range: lowest 2 - ResearchAndEngineering 0.110 (n = 29,531); highest 3 - Manufacturing 0.912 (n = 19,075).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 2 - ResearchAndEngineering | 29,531 | 136 | 0.0417 ± 0.0050 | 0.110 ± 0.191 | lowest recall |
| 5 - SalesAndMarketing | 28,459 | 64 | 0.0577 ± 0.0058 | 0.839 ± 0.024 |  |
| 3 - Manufacturing | 19,075 | 34 | 0.0663 ± 0.0014 | 0.912 ± 0.051 | highest FPR, highest recall |
| 1 - Adminstration | 15,237 | 9 | 0.0629 ± 0.0053 | 0.222 ± 0.000 |  |
| 6 - PurchasingAndContracts | 1,721 | 0 | 0.0606 ± 0.0092 | n/a (< 5 positives) |  |
| 4 - Finance | 1,357 | 0 | 0.0133 ± 0.0038 | n/a (< 5 positives) | lowest FPR |

## XGBoost (structured v3)

### all

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| all | 95,570 | 243 | 0.0559 ± 0.0017 | 0.639 ± 0.006 |  |

### role

FPR range: lowest HumanResourceSpecialist 0.000 (n = 1,192); LabManager 0.000 (n = 570); Statistician 0.000 (n = 372, small sample); HealthSafetyEngineer 0.000 (n = 322, small sample); NursePractitioner 0.000 (n = 305, small sample); Accountant 0.000 (n = 292, small sample); ComputerTrainer 0.000 (n = 230, small sample); Engineer 0.000 (n = 213, small sample); highest IndustrialEngineer 0.149 (n = 596).
Recall range: lowest HardwareEngineer 0.035 (n = 675); highest Technician 1.000 (n = 13,619); ITAdmin 1.000 (n = 9,651).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| ProductionLineWorker | 15,966 | 34 | 0.0684 ± 0.0022 | 0.971 ± 0.000 |  |
| Technician | 13,619 | 30 | 0.0736 ± 0.0010 | 1.000 ± 0.000 | highest recall |
| Salesman | 13,307 | 33 | 0.0480 ± 0.0026 | 0.434 ± 0.046 |  |
| ITAdmin | 9,651 | 5 | 0.1330 ± 0.0142 | 1.000 ± 0.000 | highest recall |
| Scientist | 3,761 | 0 | 0.0195 ± 0.0047 | n/a (< 5 positives) |  |
| SoftwareEngineer | 3,717 | 24 | 0.0029 ± 0.0018 | 0.569 ± 0.064 |  |
| AdministrativeAssistant | 3,207 | 1 | 0.0001 ± 0.0002 | n/a (< 5 positives) |  |
| ComputerScientist | 2,708 | 26 | 0.0092 ± 0.0008 | 0.833 ± 0.022 |  |
| ElectricalEngineer | 2,590 | 0 | 0.0936 ± 0.0016 | n/a (< 5 positives) |  |
| Manager | 2,491 | 2 | 0.0992 ± 0.0022 | n/a (< 5 positives) |  |
| Mathematician | 2,462 | 0 | 0.1118 ± 0.0022 | n/a (< 5 positives) |  |
| MechanicalEngineer | 2,453 | 0 | 0.0035 ± 0.0012 | n/a (< 5 positives) |  |
| ComputerProgrammer | 2,136 | 0 | 0.1261 ± 0.0019 | n/a (< 5 positives) |  |
| Physicist | 1,784 | 0 | 0.0062 ± 0.0015 | n/a (< 5 positives) |  |
| SystemsEngineer | 1,776 | 28 | 0.0008 ± 0.0009 | 0.286 ± 0.062 |  |
| TestEngineer | 1,644 | 29 | 0.0275 ± 0.0029 | 0.943 ± 0.020 |  |
| SecurityGuard | 1,558 | 2 | 0.0062 ± 0.0021 | n/a (< 5 positives) |  |
| Director | 1,262 | 0 | 0.0013 ± 0.0005 | n/a (< 5 positives) |  |
| HumanResourceSpecialist | 1,192 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| ChiefEngineer | 727 | 0 | 0.0005 ± 0.0008 | n/a (< 5 positives) |  |
| HardwareEngineer | 675 | 29 | 0.0093 ± 0.0056 | 0.035 ± 0.035 | lowest recall |
| FieldServiceEngineer | 669 | 0 | 0.0015 ± 0.0015 | n/a (< 5 positives) |  |
| VicePresident | 627 | 0 | 0.0011 ± 0.0018 | n/a (< 5 positives) |  |
| AssemblySupervisor | 596 | 0 | 0.0028 ± 0.0010 | n/a (< 5 positives) |  |
| IndustrialEngineer | 596 | 0 | 0.1488 ± 0.0079 | n/a (< 5 positives) | highest FPR |
| MaterialsEngineer | 582 | 0 | 0.0006 ± 0.0010 | n/a (< 5 positives) |  |
| LabManager | 570 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| PurchasingClerk | 481 | 0 | 0.0007 ± 0.0012 | n/a (< 5 positives) |  |
| Statistician | 372 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| Attorney | 330 | 0 | 0.0010 ± 0.0017 | n/a (< 5 positives) |  |
| HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| NursePractitioner | 305 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| Accountant | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| ComputerTrainer | 230 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| Engineer | 213 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |

### department

FPR range: lowest 1 - Personnel 0.000 (n = 992); 3 - Training 0.000 (n = 805); 2 - EmployeeRelations 0.000 (n = 595); 5 - Medical 0.000 (n = 495, small sample); 4 - Safety 0.000 (n = 432, small sample); 4 - QualityControl 0.000 (n = 422, small sample); 1 - Accounting 0.000 (n = 380, small sample); 2 - Payroll 0.000 (n = 292, small sample); 1 - MarketAnalysis 0.000 (n = 280, small sample); highest 1 - IndustrialEngineering 0.126 (n = 702).
Recall range: lowest 3 - Engineering 0.422 (n = 11,706); highest 3 - FieldService 1.000 (n = 13,904).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 3 - Assembly | 16,785 | 34 | 0.0692 ± 0.0019 | 0.971 ± 0.000 |  |
| 2 - Sales | 14,084 | 33 | 0.0477 ± 0.0025 | 0.434 ± 0.046 |  |
| 3 - FieldService | 13,904 | 30 | 0.0692 ± 0.0010 | 1.000 ± 0.000 | highest recall |
| 6 - Security | 11,728 | 9 | 0.1150 ± 0.0115 | 0.704 ± 0.064 |  |
| 3 - Engineering | 11,706 | 86 | 0.0218 ± 0.0021 | 0.422 ± 0.027 | lowest recall |
| 2 - SoftwareManagement | 8,868 | 50 | 0.0442 ± 0.0002 | 0.707 ± 0.042 |  |
| 1 - Research | 8,767 | 0 | 0.0410 ± 0.0018 | n/a (< 5 positives) |  |
| unknown | 1,387 | 0 | 0.0005 ± 0.0008 | n/a (< 5 positives) |  |
| 1 - Personnel | 992 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| 2 - ManufacturingEngineering | 952 | 0 | 0.1001 ± 0.0034 | n/a (< 5 positives) |  |
| 3 - Training | 805 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| 1 - IndustrialEngineering | 702 | 0 | 0.1263 ± 0.0067 | n/a (< 5 positives) | highest FPR |
| 2 - EmployeeRelations | 595 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| 1 - Purchasing | 584 | 0 | 0.0017 ± 0.0000 | n/a (< 5 positives) |  |
| 3 - Contracts | 540 | 0 | 0.0006 ± 0.0011 | n/a (< 5 positives) |  |
| 5 - Medical | 495 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 3 - FinancialPlanning | 489 | 0 | 0.0041 ± 0.0020 | n/a (< 5 positives) |  |
| 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Pricing | 381 | 0 | 0.0026 ± 0.0045 | n/a (< 5 positives) |  |
| 1 - Accounting | 380 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 1 - MarketAnalysis | 280 | 1 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |

### functional_unit

FPR range: lowest 6 - PurchasingAndContracts 0.001 (n = 1,721); highest 1 - Adminstration 0.089 (n = 15,237).
Recall range: lowest 2 - ResearchAndEngineering 0.527 (n = 29,531); highest 3 - Manufacturing 0.971 (n = 19,075).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 2 - ResearchAndEngineering | 29,531 | 136 | 0.0341 ± 0.0014 | 0.527 ± 0.026 | lowest recall |
| 5 - SalesAndMarketing | 28,459 | 64 | 0.0574 ± 0.0017 | 0.693 ± 0.024 |  |
| 3 - Manufacturing | 19,075 | 34 | 0.0706 ± 0.0018 | 0.971 ± 0.000 | highest recall |
| 1 - Adminstration | 15,237 | 9 | 0.0886 ± 0.0088 | 0.704 ± 0.064 | highest FPR |
| 6 - PurchasingAndContracts | 1,721 | 0 | 0.0014 ± 0.0009 | n/a (< 5 positives) | lowest FPR |
| 4 - Finance | 1,357 | 0 | 0.0015 ± 0.0007 | n/a (< 5 positives) |  |
