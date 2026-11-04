# Dataset record: CERT Insider Threat Test Dataset r4.2

## Source
- Name: CERT Insider Threat Test Dataset, release r4.2
- Producer: CERT Division, Software Engineering Institute, Carnegie Mellon University; data generated with ExactData's Dynamic Data Generator
- Downloaded from: TODO (URL of the download page)
- Date downloaded: 2026-11-04
- Archive files: r4.2.tar.bz2 (4.5 GB), answers.tar.bz2 (1.2 MB)
- Citation: TODO (copy the exact citation requested on the download page or in readme.txt)

## Licence
- ExactData end-user agreement (data/raw/r4.2/license.txt)
- Data may NOT be redistributed. The reproducibility package will include code, configs, split IDs and checksums only, not data.

## Date range
- Events: 2010-01-02 to 2011-05-17
- LDAP snapshots: monthly, 2009-12 to 2011-05 (18 files)

## Tables
| Table | Columns | Rows | First event | Last event |
|---|---|---|---|---|
| logon | id, date, user, pc, activity (Logon/Logoff) | 854,859 | 2010-01-02 06:49:00 | 2011-05-17 06:43:35 |
| device | id, date, user, pc, activity (Connect/Disconnect) | 405,380 | 2010-01-02 07:21:06 | 2011-05-16 23:22:34 |
| file | id, date, user, pc, filename, content | 445,581 | 2010-01-02 07:23:14 | 2011-05-16 23:22:34 |
| email | id, date, user, pc, to, cc, bcc, from, size, attachments, content | 2,629,979 | 2010-01-02 07:11:45 | 2011-05-16 21:16:26 |
| http | id, date, user, pc, url, content | 28,434,423 | 2010-01-02 06:55:16 | 2011-05-16 21:23:10 |
| LDAP | employee_name, user_id, email, role, business_unit, functional_unit, department, team, supervisor | 16,743 | 2009-12 | 2011-05 |
| psychometric | employee_name, user_id, O, C, E, A, N | not used | | |

Content columns (file, email, http) are stored separately in *_content.parquet with matching ids.

## Users and insiders
- Users: 1,000 (distinct user_id in LDAP)
- Malicious insiders in r4.2: 70 (from insiders.csv, dataset 4.2)
- Insiders per scenario: Scenario 1: 30, Scenario 2: 30, Scenario 3: 10
- Scenarios: 3 (answers folders r4.2-1, r4.2-2, r4.2-3); descriptions in data/raw/answers/scenarios.txt