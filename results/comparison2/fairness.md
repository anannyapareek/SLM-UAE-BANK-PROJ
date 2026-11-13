# Fairness, Comparison 2 (text) on the test split (runbook 6.4)

Logoff-only (one row per session, --rows final); alert at each seed's 5%-FPR threshold from the validation final-row eval. Mean ± std over 3 seeds. Groups: the pre-declared fairness proxies role, department and functional_unit (tenure was dropped on 2026-11-04 and is not available). Shown only for groups with >= 200 sessions; recall only where a group has >= 5 positive sessions. Flags: the lowest and highest FPR (and recall) in each attribute; **small sample** marks such an extreme from a group with fewer than 500 sessions.

## SLM, text (Qwen2.5-1.5B LoRA)

### all

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| all | 95,570 | 243 | 0.0589 ± 0.0004 | 0.840 ± 0.022 |  |

### role

FPR range: lowest Director 0.000 (n = 1,262); LabManager 0.000 (n = 570); HealthSafetyEngineer 0.000 (n = 322, small sample); NursePractitioner 0.000 (n = 305, small sample); Accountant 0.000 (n = 292, small sample); Engineer 0.000 (n = 213, small sample); highest HardwareEngineer 0.189 (n = 675).
Recall range: lowest HardwareEngineer 0.586 (n = 675); highest ProductionLineWorker 1.000 (n = 15,966).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| ProductionLineWorker | 15,966 | 34 | 0.0489 ± 0.0037 | 1.000 ± 0.000 | highest recall |
| Technician | 13,619 | 30 | 0.0534 ± 0.0063 | 0.967 ± 0.033 |  |
| Salesman | 13,307 | 33 | 0.0408 ± 0.0017 | 0.869 ± 0.063 |  |
| ITAdmin | 9,651 | 5 | 0.0915 ± 0.0063 | 0.800 ± 0.000 |  |
| Scientist | 3,761 | 0 | 0.0653 ± 0.0027 | n/a (< 5 positives) |  |
| SoftwareEngineer | 3,717 | 24 | 0.0332 ± 0.0004 | 0.958 ± 0.000 |  |
| AdministrativeAssistant | 3,207 | 1 | 0.0183 ± 0.0067 | n/a (< 5 positives) |  |
| ComputerScientist | 2,708 | 26 | 0.0454 ± 0.0018 | 0.885 ± 0.000 |  |
| ElectricalEngineer | 2,590 | 0 | 0.0356 ± 0.0054 | n/a (< 5 positives) |  |
| Manager | 2,491 | 2 | 0.0940 ± 0.0123 | n/a (< 5 positives) |  |
| Mathematician | 2,462 | 0 | 0.0988 ± 0.0038 | n/a (< 5 positives) |  |
| MechanicalEngineer | 2,453 | 0 | 0.0737 ± 0.0015 | n/a (< 5 positives) |  |
| ComputerProgrammer | 2,136 | 0 | 0.0999 ± 0.0027 | n/a (< 5 positives) |  |
| Physicist | 1,784 | 0 | 0.1362 ± 0.0037 | n/a (< 5 positives) |  |
| SystemsEngineer | 1,776 | 28 | 0.0343 ± 0.0060 | 0.821 ± 0.036 |  |
| TestEngineer | 1,644 | 29 | 0.0809 ± 0.0092 | 0.667 ± 0.072 |  |
| SecurityGuard | 1,558 | 2 | 0.0718 ± 0.0077 | n/a (< 5 positives) |  |
| Director | 1,262 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| HumanResourceSpecialist | 1,192 | 0 | 0.0333 ± 0.0049 | n/a (< 5 positives) |  |
| ChiefEngineer | 727 | 0 | 0.0830 ± 0.0147 | n/a (< 5 positives) |  |
| HardwareEngineer | 675 | 29 | 0.1889 ± 0.0067 | 0.586 ± 0.000 | highest FPR, lowest recall |
| FieldServiceEngineer | 669 | 0 | 0.0538 ± 0.0068 | n/a (< 5 positives) |  |
| VicePresident | 627 | 0 | 0.0579 ± 0.0120 | n/a (< 5 positives) |  |
| AssemblySupervisor | 596 | 0 | 0.0291 ± 0.0042 | n/a (< 5 positives) |  |
| IndustrialEngineer | 596 | 0 | 0.1186 ± 0.0137 | n/a (< 5 positives) |  |
| MaterialsEngineer | 582 | 0 | 0.0624 ± 0.0114 | n/a (< 5 positives) |  |
| LabManager | 570 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| PurchasingClerk | 481 | 0 | 0.1275 ± 0.0295 | n/a (< 5 positives) |  |
| Statistician | 372 | 0 | 0.1613 ± 0.0027 | n/a (< 5 positives) |  |
| Attorney | 330 | 0 | 0.0091 ± 0.0000 | n/a (< 5 positives) |  |
| HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| NursePractitioner | 305 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| Accountant | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| ComputerTrainer | 230 | 0 | 0.1754 ± 0.0100 | n/a (< 5 positives) |  |
| Engineer | 213 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |

### department

FPR range: lowest 5 - Medical 0.000 (n = 495, small sample); 4 - Safety 0.000 (n = 432, small sample); 4 - QualityControl 0.000 (n = 422, small sample); 1 - Accounting 0.000 (n = 380, small sample); 2 - Payroll 0.000 (n = 292, small sample); highest 2 - Pricing 0.158 (n = 381, small sample).
Recall range: lowest 6 - Security 0.667 (n = 11,728); highest 3 - Assembly 1.000 (n = 16,785).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 3 - Assembly | 16,785 | 34 | 0.0504 ± 0.0037 | 1.000 ± 0.000 | highest recall |
| 2 - Sales | 14,084 | 33 | 0.0401 ± 0.0025 | 0.869 ± 0.063 |  |
| 3 - FieldService | 13,904 | 30 | 0.0498 ± 0.0057 | 0.967 ± 0.033 |  |
| 6 - Security | 11,728 | 9 | 0.0877 ± 0.0063 | 0.667 ± 0.000 | lowest recall |
| 3 - Engineering | 11,706 | 86 | 0.0639 ± 0.0018 | 0.690 ± 0.034 |  |
| 2 - SoftwareManagement | 8,868 | 50 | 0.0683 ± 0.0021 | 0.920 ± 0.000 |  |
| 1 - Research | 8,767 | 0 | 0.0835 ± 0.0013 | n/a (< 5 positives) |  |
| unknown | 1,387 | 0 | 0.0262 ± 0.0054 | n/a (< 5 positives) |  |
| 1 - Personnel | 992 | 0 | 0.0528 ± 0.0150 | n/a (< 5 positives) |  |
| 2 - ManufacturingEngineering | 952 | 0 | 0.0287 ± 0.0112 | n/a (< 5 positives) |  |
| 3 - Training | 805 | 0 | 0.1023 ± 0.0026 | n/a (< 5 positives) |  |
| 1 - IndustrialEngineering | 702 | 0 | 0.1007 ± 0.0116 | n/a (< 5 positives) |  |
| 2 - EmployeeRelations | 595 | 0 | 0.0151 ± 0.0133 | n/a (< 5 positives) |  |
| 1 - Purchasing | 584 | 0 | 0.1050 ± 0.0243 | n/a (< 5 positives) |  |
| 3 - Contracts | 540 | 0 | 0.0056 ± 0.0000 | n/a (< 5 positives) |  |
| 5 - Medical | 495 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 3 - FinancialPlanning | 489 | 0 | 0.0136 ± 0.0077 | n/a (< 5 positives) |  |
| 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Pricing | 381 | 0 | 0.1575 ± 0.0026 | n/a (< 5 positives) | highest FPR, **small sample** (n < 500) |
| 1 - Accounting | 380 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 1 - MarketAnalysis | 280 | 1 | 0.0119 ± 0.0041 | n/a (< 5 positives) |  |

### functional_unit

FPR range: lowest 4 - Finance 0.005 (n = 1,357); highest 1 - Adminstration 0.079 (n = 15,237).
Recall range: lowest 1 - Adminstration 0.667 (n = 15,237); highest 3 - Manufacturing 1.000 (n = 19,075).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 2 - ResearchAndEngineering | 29,531 | 136 | 0.0707 ± 0.0006 | 0.774 ± 0.021 |  |
| 5 - SalesAndMarketing | 28,459 | 64 | 0.0443 ± 0.0030 | 0.917 ± 0.039 |  |
| 3 - Manufacturing | 19,075 | 34 | 0.0495 ± 0.0031 | 1.000 ± 0.000 | highest recall |
| 1 - Adminstration | 15,237 | 9 | 0.0793 ± 0.0063 | 0.667 ± 0.000 | highest FPR, lowest recall |
| 6 - PurchasingAndContracts | 1,721 | 0 | 0.0722 ± 0.0085 | n/a (< 5 positives) |  |
| 4 - Finance | 1,357 | 0 | 0.0049 ± 0.0028 | n/a (< 5 positives) | lowest FPR |

## XGBoost, text

### all

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| all | 95,570 | 243 | 0.0600 ± 0.0017 | 0.722 ± 0.026 |  |

### role

FPR range: lowest AssemblySupervisor 0.000 (n = 596); LabManager 0.000 (n = 570); HealthSafetyEngineer 0.000 (n = 322, small sample); Accountant 0.000 (n = 292, small sample); Engineer 0.000 (n = 213, small sample); highest ITAdmin 0.243 (n = 9,651).
Recall range: lowest HardwareEngineer 0.264 (n = 675); highest ProductionLineWorker 1.000 (n = 15,966); Technician 1.000 (n = 13,619); ITAdmin 1.000 (n = 9,651).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| ProductionLineWorker | 15,966 | 34 | 0.0648 ± 0.0036 | 1.000 ± 0.000 | highest recall |
| Technician | 13,619 | 30 | 0.0510 ± 0.0026 | 1.000 ± 0.000 | highest recall |
| Salesman | 13,307 | 33 | 0.0255 ± 0.0035 | 0.697 ± 0.080 |  |
| ITAdmin | 9,651 | 5 | 0.2431 ± 0.0106 | 1.000 ± 0.000 | highest FPR, highest recall |
| Scientist | 3,761 | 0 | 0.0256 ± 0.0078 | n/a (< 5 positives) |  |
| SoftwareEngineer | 3,717 | 24 | 0.0009 ± 0.0007 | 0.722 ± 0.251 |  |
| AdministrativeAssistant | 3,207 | 1 | 0.0020 ± 0.0005 | n/a (< 5 positives) |  |
| ComputerScientist | 2,708 | 26 | 0.0055 ± 0.0021 | 0.500 ± 0.038 |  |
| ElectricalEngineer | 2,590 | 0 | 0.1106 ± 0.0040 | n/a (< 5 positives) |  |
| Manager | 2,491 | 2 | 0.0832 ± 0.0011 | n/a (< 5 positives) |  |
| Mathematician | 2,462 | 0 | 0.1060 ± 0.0134 | n/a (< 5 positives) |  |
| MechanicalEngineer | 2,453 | 0 | 0.0117 ± 0.0017 | n/a (< 5 positives) |  |
| ComputerProgrammer | 2,136 | 0 | 0.0757 ± 0.0143 | n/a (< 5 positives) |  |
| Physicist | 1,784 | 0 | 0.0125 ± 0.0044 | n/a (< 5 positives) |  |
| SystemsEngineer | 1,776 | 28 | 0.0114 ± 0.0015 | 0.679 ± 0.036 |  |
| TestEngineer | 1,644 | 29 | 0.0194 ± 0.0068 | 0.839 ± 0.053 |  |
| SecurityGuard | 1,558 | 2 | 0.0216 ± 0.0035 | n/a (< 5 positives) |  |
| Director | 1,262 | 0 | 0.0005 ± 0.0005 | n/a (< 5 positives) |  |
| HumanResourceSpecialist | 1,192 | 0 | 0.0025 ± 0.0008 | n/a (< 5 positives) |  |
| ChiefEngineer | 727 | 0 | 0.0083 ± 0.0060 | n/a (< 5 positives) |  |
| HardwareEngineer | 675 | 29 | 0.0330 ± 0.0103 | 0.264 ± 0.080 | lowest recall |
| FieldServiceEngineer | 669 | 0 | 0.0055 ± 0.0023 | n/a (< 5 positives) |  |
| VicePresident | 627 | 0 | 0.0165 ± 0.0056 | n/a (< 5 positives) |  |
| AssemblySupervisor | 596 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| IndustrialEngineer | 596 | 0 | 0.1091 ± 0.0191 | n/a (< 5 positives) |  |
| MaterialsEngineer | 582 | 0 | 0.0023 ± 0.0040 | n/a (< 5 positives) |  |
| LabManager | 570 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR |
| PurchasingClerk | 481 | 0 | 0.0014 ± 0.0012 | n/a (< 5 positives) |  |
| Statistician | 372 | 0 | 0.0197 ± 0.0138 | n/a (< 5 positives) |  |
| Attorney | 330 | 0 | 0.0313 ± 0.0070 | n/a (< 5 positives) |  |
| HealthSafetyEngineer | 322 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| NursePractitioner | 305 | 0 | 0.0011 ± 0.0019 | n/a (< 5 positives) |  |
| Accountant | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| ComputerTrainer | 230 | 0 | 0.0072 ± 0.0066 | n/a (< 5 positives) |  |
| Engineer | 213 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |

### department

FPR range: lowest 4 - Safety 0.000 (n = 432, small sample); 4 - QualityControl 0.000 (n = 422, small sample); 2 - Payroll 0.000 (n = 292, small sample); highest 6 - Security 0.207 (n = 11,728).
Recall range: lowest 3 - Engineering 0.593 (n = 11,706); highest 3 - Assembly 1.000 (n = 16,785); 3 - FieldService 1.000 (n = 13,904).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 3 - Assembly | 16,785 | 34 | 0.0650 ± 0.0041 | 1.000 ± 0.000 | highest recall |
| 2 - Sales | 14,084 | 33 | 0.0260 ± 0.0036 | 0.697 ± 0.080 |  |
| 3 - FieldService | 13,904 | 30 | 0.0447 ± 0.0027 | 1.000 ± 0.000 | highest recall |
| 6 - Security | 11,728 | 9 | 0.2074 ± 0.0083 | 0.667 ± 0.000 | highest FPR |
| 3 - Engineering | 11,706 | 86 | 0.0337 ± 0.0015 | 0.593 ± 0.012 | lowest recall |
| 2 - SoftwareManagement | 8,868 | 50 | 0.0276 ± 0.0029 | 0.607 ± 0.141 |  |
| 1 - Research | 8,767 | 0 | 0.0433 ± 0.0074 | n/a (< 5 positives) |  |
| unknown | 1,387 | 0 | 0.0075 ± 0.0025 | n/a (< 5 positives) |  |
| 1 - Personnel | 992 | 0 | 0.0030 ± 0.0010 | n/a (< 5 positives) |  |
| 2 - ManufacturingEngineering | 952 | 0 | 0.0875 ± 0.0082 | n/a (< 5 positives) |  |
| 3 - Training | 805 | 0 | 0.0112 ± 0.0022 | n/a (< 5 positives) |  |
| 1 - IndustrialEngineering | 702 | 0 | 0.0926 ± 0.0162 | n/a (< 5 positives) |  |
| 2 - EmployeeRelations | 595 | 0 | 0.0006 ± 0.0010 | n/a (< 5 positives) |  |
| 1 - Purchasing | 584 | 0 | 0.0023 ± 0.0010 | n/a (< 5 positives) |  |
| 3 - Contracts | 540 | 0 | 0.0278 ± 0.0111 | n/a (< 5 positives) |  |
| 5 - Medical | 495 | 0 | 0.0007 ± 0.0012 | n/a (< 5 positives) |  |
| 3 - FinancialPlanning | 489 | 0 | 0.0136 ± 0.0031 | n/a (< 5 positives) |  |
| 4 - Safety | 432 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 4 - QualityControl | 422 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 2 - Pricing | 381 | 0 | 0.0219 ± 0.0084 | n/a (< 5 positives) |  |
| 1 - Accounting | 380 | 0 | 0.0009 ± 0.0015 | n/a (< 5 positives) |  |
| 2 - Payroll | 292 | 0 | 0.0000 ± 0.0000 | n/a (< 5 positives) | lowest FPR, **small sample** (n < 500) |
| 1 - MarketAnalysis | 280 | 1 | 0.0072 ± 0.0072 | n/a (< 5 positives) |  |

### functional_unit

FPR range: lowest 4 - Finance 0.005 (n = 1,357); highest 1 - Adminstration 0.161 (n = 15,237).
Recall range: lowest 2 - ResearchAndEngineering 0.598 (n = 29,531); highest 3 - Manufacturing 1.000 (n = 19,075).

| Group | Sessions | Positives | FPR | Recall | Flag |
|---|---|---|---|---|---|
| 2 - ResearchAndEngineering | 29,531 | 136 | 0.0345 ± 0.0016 | 0.598 ± 0.048 | lowest recall |
| 5 - SalesAndMarketing | 28,459 | 64 | 0.0348 ± 0.0025 | 0.844 ± 0.041 |  |
| 3 - Manufacturing | 19,075 | 34 | 0.0649 ± 0.0039 | 1.000 ± 0.000 | highest recall |
| 1 - Adminstration | 15,237 | 9 | 0.1611 ± 0.0065 | 0.667 ± 0.000 | highest FPR |
| 6 - PurchasingAndContracts | 1,721 | 0 | 0.0143 ± 0.0048 | n/a (< 5 positives) |  |
| 4 - Finance | 1,357 | 0 | 0.0052 ± 0.0007 | n/a (< 5 positives) | lowest FPR |
