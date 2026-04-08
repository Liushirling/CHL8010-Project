# Depression Risk in Asthma vs. COPD vs. ACO
### A Longitudinal Analysis Using CPCSSN Data

This repository contains the data wrangling pipeline for encounter, patient, lab, and family history datasets from the **CPCSSN** database. The pipeline is designed to process complex, "dirty" EHR data, perform cohort filtering for survival analysis, and generate survival curves with associated log-rank tests.

## Installation

Follow these instructions to set up your local environment:

```bash
# Clone the repository
git clone [https://github.com/Liushirling/CHL8010-Project.git](https://github.com/Liushirling/CHL8010-Project.git)
cd CHL8010-Project

# Install dependencies
pip install -r requirements.txt
```

## Project Structures

```Plaintext
├── src/
│   ├── main.py                    # Pipeline entry point and execution script
│   ├── data_input.py              # Data ingestion and localized .pkl caching
│   ├── data_cleaning.py           # Data standardization and ICD-9 code mapping
│   ├── filter_target_population.py # Cohort construction and covariate extraction
│   └── survival_model.py          # Kaplan-Meier estimation, log-rank testing, and Cox modeling
├── raw/                          # Original EHR datasets
│   ├── output/                   # Generated cleaned and analytical datasets
└── requirements.txt               # Python environment dependencies
└── README.md               # Project overview and execution instructions
└── data_schema_standardization.txt  # Data schema documentation
```

## Usage

Run the full pipeline using the following command:

```bash
python src/main.py
```

## Features / Pipeline Logic

### **Data Input**
  - Automatically detect and load CSV/Pickle files, cache cleaned data (saving clean datasets as `.csv` and `.pkl`).
  - Seamlessly process large files with chunk-based parsing, automatically detect parsing errors.
  - Features automatic "bad line" recovery, which can isolate damaged lines, remove invalid escape characters/quotes, repair misaligned delimiters, and merge the recovered data back into the main dataset.

### **Data Cleaning**
  - Cleans column headers and strips string quotes.
  - Standardizes pseudo-missing data types into `pd.NA`.
  - Identify and remove clinically inappropriate dates.
  - Determine valid value ranges (convert abnormal negative test results to `NA`).
  - Binarization of `Sex` variable and extracts `ICD3` categories from `ICD9` codes (e.g., 490.3 -> 490).
  - Removes duplicates.

### **Filter Target Population**
  - Establishes a flexible `STUDY CONFIGURATION` and `DATASET SCHEMA` to easily map columns by index and swap out study exposures or ICD-9 codes without altering core functions.
  - Filters patiens into distinct exposure groups based on their specific ICD-9 diagnosis history.
  - Determines a patient-specific `index_date` based on the sequence of their diagnoses, dropping patients with prior events to ensure incident risk tracking
  - Computes the patient's exact age at their index dates and calculates the duration of follow-up (time to event or study censor date).

### **Survival Analysis**
  - Performs right-censored time-to-event modeling to evaluate the incidence risk of depression.
  - **Kaplan-Meier Estimator**: Plots survival curves for the Asthma, COPD, and ACO cohorts with confidence intervals.
  - **Log-Rank Testing**: Conducts pairwise statistical tests between the three exposure groups to assess the differences in survival probabilities.
  - **Confounder Assessment**: Adjusts percentage change in log hazard ratios for relevant baseline covariates.


## Final Dataset Schema

The `filter_target_population` module produces a processed patient-level DataFrame with the following columns:

| Column | Description |
| :--- | :--- |
| **Patient_ID** | Unique identifier (string) |
| **age** | Exact age calculated at the `index_date` |
| **Sex** | Binary indicator: `1` for Male, `0` for Female or Other |
| **BirthYear** | The recorded year of birth for the patient|
| **dx_dict** | A dictionary mapping clinical conditions to their chronological diagnosis dates |
| **exposure** | Group assignment (`asthma` / `copd` / `aco`) |
| **index_date** | Baseline date for the study; the date of the qualifying exposure diagnosis |
| **first_asthma** | Date of the first recorded Asthma diagnosis |
| **second_asthma** | Date of the second recorded Asthma diagnosis (required for ACO group logic) |
| **first_copd** | Date of the first recorded COPD diagnosis | 
| **depression_date** | Date of the first recorded Depression diagnosis (the study event)|
| **FamilyRelationship** | Semicolon-separated list of relatives with a recorded medical history |
| **FamilyDiagnosis** | Semicolon-separated list of ICD-9 codes found in the patient's family history |
| **FamilyDescription** | Semicolon-separated clinical descriptions of family history diagnoses |
| **diabetes** | Binary (0/1) based on baseline Glucose/HbA1c clinical thresholds |
| **Family_History** | Binary (0/1) for family history of depression |
| **Num_Comorbidities** | Count of unique baseline diagnoses (excluding primary study codes) |
| **event** | Binary (0/1) indicating a Depression diagnosis during follow-up |
| **end_date** | Follow-up termination date: `depression_date` or `censor_date`|
| **duration** | Follow-up time in days (from `index_date` to `event_date` or `censor_date`) |
