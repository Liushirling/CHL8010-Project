# Depression Risk in Asthma vs. COPD vs. ACO
### A Longitudinal Analysis Using CPCSSN Data

This repository contains the data wrangling pipeline for encounter, patient, lab, and family history datasets from the **CPCSSN** database. The pipeline is designed to process complex, "dirty" EHR data, perform cohort filtering for survival analysis, and generate survival curves with associated log-rank tests.

## Installation

Follow these instructions to set up your local environment:

```bash
# Clone the repository
git clone [https://github.com/Liushirling/CHL8010-Project.git](https://github.com/Liushirling/CHL8010-Project.git)
cd CHL8010-Project

# Create and activate the environment
conda create --name CHL8010_env python=3.9
conda activate CHL8010_env

# Install dependencies
pip install -r requirements.txt
```

## Project Structures

```Plaintext
├── src/
│   ├── main.py                    # Entry point of the pipeline
│   ├── data_input.py              # Initial data loading and .pkl caching
│   ├── data_cleaning.py           # Pre-processing and ICD-9 mapping
│   ├── filter_target_population.py # Cohort construction & covariates processing
│   └── survival_model.py          # Statistical modeling, Kaplan-Meier plotting, and Log rank test
├── raw/                          # Local storage (excluded via .gitignore)
│   ├── output/                   # Output storage (excluded via .gitignore)
└── requirements.txt               # Python dependencies
└── README.md               # Python dependencies
└── data_schema_standardization.txt  # Data schema documentation
```

## Usage

Run the full pipeline using the following command:
```bash
python src/main.py
```

## Features / Pipeline Logic

- Data Input
  - Efficiently loads CSV/Pickle files with automated caching.

- Data Cleaning
  - Cleaning dataset and standardization.

- Cohort Construction
  - Filters patients by ICD-9 codes to identify Asthma, COPD, and ACO (Asthma-COPD Overlap) groups.
  - Calculates baseline comorbidities and processes lab results (Glucose/HbA1c).

- Survival Analysis
  - Performs time-to-event modeling for Depression risk.
  - Kaplan-Meier plotting
  - Log rank test

## Final Dataset Schema

The `filter_target_population` module produces a processed patient-level DataFrame with the following columns:

| Column | Description |
| :--- | :--- |
| **Patient_ID** | Unique identifier (string) |
| **Sex** | Binary indicator: `1` for Male, `0` for Female or Other |
| **BirthYear** | The recorded year of birth for the patient|
| **BirthMonth** | The recorded month of birth for the patient.|
| **Optedout** | Binary indicator (0/1) for patients who opted out of data sharing|
| **OptOutDate** | The specific date a patient recorded their opt-out status|
| **dx_dict** | A dictionary mapping clinical conditions to their chronological diagnosis dates |
| **first_asthma** | Date of the first recorded Asthma diagnosis |
| **second_asthma** | Date of the second recorded Asthma diagnosis (required for ACO group logic) |
| **first_copd** | Date of the first recorded COPD diagnosis |
| **depression_date** | Date of the first recorded Depression diagnosis (the study event)|
| **index_date** | Baseline date for the study; the date of the qualifying exposure diagnosis |
| **age** | Exact age calculated at the `index_date` |
| **exposure** | Group assignment (`asthma` / `copd` / `aco`) |
| **event** | Binary (0/1) indicating a Depression diagnosis during follow-up |
| **end_date** | Follow-up termination date: `depression_date` or `censor_date` (2015-07-21)|
| **duration** | Follow-up time in days (from `index_date` to `event_date` or `censor_date`) |
| **Num_Comorbidities** | Count of unique baseline diagnoses (excluding primary study codes) |
| **diabetes** | Binary (0/1) based on baseline Glucose/HbA1c clinical thresholds |
| **FamilyRelationship** | Semicolon-separated list of relatives with a recorded medical history |
| **FamilyDiagnosis** | Semicolon-separated list of ICD-9 codes found in the patient's family history |
| **FamilyDescription** | Semicolon-separated clinical descriptions of family history diagnoses |
| **Family_History** | Binary (0/1) for family history of depression |
