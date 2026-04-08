import pandas as pd
import numpy as np
import os

BASE_PATH = "../raw/output"
encounter_dx_path = os.path.join(BASE_PATH, "encounter_dx_cleaned.csv")
patient_path = os.path.join(BASE_PATH, "patient_cleaned.csv")
lab_path: str = "../raw/output/lab_cleaned.csv"
family_path: str = "../raw/output/family_cleaned.csv"
censor_date = pd.to_datetime('2015-07-21')

# ---------- STUDY CONFIGURATION --------------

# condition code
exposure1_code = [493]
exposure2_code = [490, 491, 492, 494, 495, 496]

# conditions
exposure1 = "asthma"
exposure2 = "copd"
exposure3 = "aco"

# exposure date
first_exposure1 = f"first_{exposure1}"
second_exposure1 = f"second_{exposure1}"
first_exposure2 = f"first_{exposure2}"

# event
event = "depression"
event_code = [296, 300, 309, 311]
event_date = f"{event}_date"

# covariates
# 1. lab
test1_key, test1_val = "glucose", "Glucose"
test2_key, test2_val = "hba1c", "HbA1c"
threshold1 = 7.0
threshold2 = 6.5
covariates1 = "diabetes"

# 2. family history
covariates2 = "Family_History"

# 3. comorbidities
covariates3 = "Num_Comorbidities"
study_codes = [490, 491, 492, 493, 494, 495, 496, 296, 300, 309, 311]

# ------------ DATASET SCHEMA (COLUMN INDICES) --------
# Please refer to data_schema_standardization.txt

# encounter_dx columns
enc_id = 3 # Patient_ID
enc_dx_text = 6
enc_dx_type = 9
enc_dx_code = 11
enc_date = 12 # DateCreated
enc_icd3_num = 14

# lab columns
lab_id = 3 # Patient_ID
lab_name = 7
lab_result = 13
lab_date = 20 # DateCreated

# patient columns
pat_id = 0 # Patient_ID
pat_sex = 1
pat_birth = 2

# family columns
fam_id = 3 # Patient_ID
fam_text = 6
fam_code = 10
fam_rel = 12

# ---------------------------------------------------------------------

def load_data(file_path, dtype=str, sep="|", engine="python",
              quoting=0, on_bad_lines="skip"):
    """
    General loader: prefer .pkl, otherwise load .csv and cache as .pkl
    """
    pickle_path = file_path.replace(".csv", ".pkl")
    
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)
    elif os.path.exists(file_path):
        df = pd.read_csv(
            file_path,
            dtype=dtype,
            sep=sep,
            engine=engine,
            quoting=quoting,
            on_bad_lines=on_bad_lines
        )
        df.to_pickle(pickle_path)
        return df
    else:
        return pd.DataFrame()

def add_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates ICD9 diagnosis codes to calssify encounters as the event and exposures.

    Uses indices from encounter_dx:
        [9] enc_dx_type (DiagnosisCodeType_calc)
        [14] enc_icd3_num (ICD3_num)

    Output:
        The original dataframe with a new "condition" column appended at the end (index 15).
    """
    # Check if empty or missinf required indices
    required_max_index = max(enc_dx_type, enc_icd3_num)
    if df is None or df.empty or df.shape[1] <= required_max_index:
        safe_df = df.copy() if df is not None else pd.DataFrame()
        safe_df["condition"] = pd.Series(dtype='object')
        return safe_df
    
    col_dx_type = df.columns[enc_dx_type]
    col_icd3_num = df.columns[enc_icd3_num]
    df["condition"] = pd.Series(dtype='object')

    mask_icd9 = df[col_dx_type] == "ICD9"

    # Event
    df.loc[
        mask_icd9 & df[col_icd3_num].isin(event_code),
        "condition"
    ] = event

    # Exposure 1
    df.loc[
        mask_icd9 &
        df["condition"].isna() &
        df[col_icd3_num].isin(exposure1_code),
        "condition"
    ] = exposure1

    # Exposure 2
    df.loc[
        mask_icd9 &
        df["condition"].isna() &
        df[col_icd3_num].isin(exposure2_code),
        "condition"
    ] = exposure2

    return df

def build_patient_dx_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates a patient's valid diagnosis dates into a dictionary grouped by condition type.
    
    Uses indices from encounter_dx (modified):
        [3] enc_id (Patient_ID)
        [12] enc_date (DateCreated)

    Output:
        A dataframe with:
        - Patient_ID
        - dx_dict: {condition: [datetime]}
    """
    # Check if empty, missing required indices, or missing the 'condition' column
    required_max_index = max(enc_id, enc_date)
    if df is None or df.empty or df.shape[1] <= required_max_index or "condition" not in df.columns:
        return pd.DataFrame(columns=["Patient_ID", "dx_dict"])
    
    col_id = df.columns[enc_id]
    col_date = df.columns[enc_date]

    df_valid = df[df["condition"].notna()].copy()

    df_valid[col_date] = pd.to_datetime(
        df_valid[col_date],
        errors="coerce"
    )

    df_valid = df_valid[df_valid[col_date].notna()]

    def build_dict(group):
        dx_dict = {}

        for _, row in group.iterrows():
            cond = row["condition"]
            date = row[col_date]  

            if cond not in dx_dict:
                dx_dict[cond] = []

            dx_dict[cond].append(date)

        for k in dx_dict:
            dx_dict[k] = sorted(dx_dict[k])

        return dx_dict

    result = (
        df_valid
        .groupby(col_id)
        .apply(build_dict)
        .reset_index(name="dx_dict")
    )
    
    # safely rename the grouped column back to Patient_ID
    result.rename(columns={result.columns[0]: "Patient_ID"}, inplace=True)

    return result

def merge_patient_dx(patient_df: pd.DataFrame, 
                     patient_dx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the patient demographics table with created diagnosis dictionary.

    Uses indices from patient_df:
        [0] pat_id (Patient_ID)

    Output:
        patient_df + dx_dict column
    """
    pat_col = patient_df.columns[pat_id]
    patient_df[pat_col] = patient_df[pat_col].astype(str)
    
    dx_col = patient_dx_df.columns[0]
    patient_dx_df[dx_col] = patient_dx_df[dx_col].astype(str)

    merged = patient_df.merge(
        patient_dx_df,
        left_on=pat_col,
        right_on=dx_col,
        how="inner"   
    )

    return merged

# construct cohort
def build_exposure_groups(patient_with_dx: pd.DataFrame):
    """
    Categorizes patients into 3 distinct exposure groups based on the chronology of their diagnoses.
    Extracts chronological dates and calculates exact age.
    
    Input:
        patient_with_dx
        
    Uses indices from patient_with_dx:
        [2] pat_birth (BirthYear)
        
    Output::
        - index_date (date of first exposure 1/2 or 2nd exposure 1 for group 3)
        - event_date (date of first lifetime event)
        - age (patient's exact age calculated at the index_date)
    """
    
    df = patient_with_dx.copy()

    def get_nth_date(d, condition, n):
        if not isinstance(d, dict) or condition not in d:
            return pd.NaT
            
        dates = sorted(d[condition])
        
        return dates[n] if len(dates) > n else pd.NaT

    df[first_exposure1] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, exposure1, 0))
    
    df[second_exposure1] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, exposure1, 1))
    
    df[first_exposure2] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, exposure2, 0))
    
    df[event_date] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, event, 0))


    # 1. Exposure 3 Group
    exposure3_mask = (
        df[first_exposure2].notna() &
        df[second_exposure1].notna() &
        (df[first_exposure2] < df[first_exposure1]))
    exposure3_group = df[exposure3_mask].copy()
    exposure3_group["index_date"] = exposure3_group[first_exposure1] # Exposure 3 index date is 2nd exposure1

    # exclude if event occurs on/before index date
    exposure3_group = exposure3_group[
        exposure3_group[event_date].isna() |
        ~(exposure3_group[event_date] <= exposure3_group["index_date"])]

    # 2. Exposure 2 Group
    exposure2_mask = (
        df[first_exposure2].notna() &
        (df[first_exposure2] < df[first_exposure1].fillna(pd.Timestamp.max)) &
        ~exposure3_mask)
    exposure2_group = df[exposure2_mask].copy()
    exposure2_group["index_date"] = exposure2_group[first_exposure2]

    # exclude if event occurs on/before index date
    exposure2_group = exposure2_group[
        exposure2_group[event_date].isna() |
        (exposure2_group[event_date] > exposure2_group["index_date"])]

    # 3. Exposure 1 Group
    exposure1_mask = (
        df[first_exposure1].notna() &
        (df[first_exposure1] <= df[first_exposure2].fillna(pd.Timestamp.max)) &
        ~exposure3_mask)
    exposure1_group = df[exposure1_mask].copy()
    exposure1_group["index_date"] = exposure1_group[first_exposure1]

    # exclude if event occurs on/before index date
    exposure1_group = exposure1_group[
        exposure1_group[event_date].isna() |
        (exposure1_group[event_date] > exposure1_group["index_date"])]

    # calculate age based on the index dates
    def calc_age(group_df):
        group_df.iloc[:, pat_birth] = pd.to_numeric(group_df.iloc[:, pat_birth], errors = "coerce")
        group_df["age"] = group_df["index_date"].dt.year - group_df.iloc[:, pat_birth]
        return group_df

    return calc_age(exposure1_group), calc_age(exposure2_group), calc_age(exposure3_group)

# construct lab data
def process_lab_data(labs_df: str, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Isolates baseline lab results prior to the patient's index date.
    
    Input:
        labs_df, final_cohort
        
    Uses indices from lab:
        [3] lab_id (Patient_ID)
        [7] lab_name (Name_orig)
        [13] lab_result (TestResult_orig)
        [20] lab_date (DateCreated)
        
    Output:
        - Test (Glucose, HbA1c, or Other)
        - Test_Results (lab result strictly prior to index_date, "Unknown" if no baseline lab exists)
    """
    # Check if empty OR if the dataset is missing the required indices
    required_max_index = max(lab_id, lab_name, lab_result, lab_date)
    if labs_df is None or labs_df.empty or labs_df.shape[1] < required_max_index:
        return final_cohort

    labs = labs_df.copy()

    col_id = labs.columns[lab_id]
    col_name = labs.columns[lab_name]
    col_result = labs.columns[lab_result]
    col_date = labs.columns[lab_date]
    
    labs[col_id] = labs[col_id].astype(str)

    merged_labs = labs.merge(
        final_cohort[["Patient_ID", "index_date"]],
        left_on=col_id, 
        right_on="Patient_ID",
        how="inner")
    
    merged_labs["ObservationDate"] = pd.to_datetime(merged_labs[col_date], errors="coerce")

    baseline_labs = merged_labs[merged_labs["ObservationDate"] <= merged_labs["index_date"]].copy()
    
    numeric_results = pd.to_numeric(baseline_labs[col_result], errors='coerce')
    baseline_labs["Test_Results"] = numeric_results.fillna('Unknown')

    substring_to_standard = {
        test1_key: test1_val,
        test2_key: test2_val
    }

    def standardize_name(name):
        if pd.isna(name) or str(name).strip() == "nan" or str(name).strip() == "": 
            return np.nan
            
        name_lower = str(name).lower()
        for sub, std in substring_to_standard.items():
            if sub in name_lower:
                return std
        return "Other"

    # Apply standardization
    baseline_labs["Test"] = baseline_labs[col_name].apply(standardize_name)
    baseline_labs = baseline_labs.dropna(subset=["Test"])

    # sort and drop duplicates
    baseline_labs = baseline_labs.sort_values(by=["Patient_ID", "Test", "ObservationDate"])
    recent_labs = baseline_labs.drop_duplicates(subset=["Patient_ID", "Test"], keep="last")

    # keep necessary columns & merge back to the main cohort
    long_format_labs = recent_labs[["Patient_ID", "Test", "Test_Results"]]
    merged_final = final_cohort.merge(long_format_labs, on="Patient_ID", how="left")
    merged_final["Test_Results"] = merged_final["Test_Results"].fillna("Unknown")
    
    return merged_final

# Build covariates1 by Test and Test_Result 
def build_covariates1_patient_df(df):
    """
    Applies clinical thresholds to lab test results to determine disease presence.
    
    Input：
        df: dataframe, includes 
            - Patient_ID
            - Test
            - Test_Results
    
    Uses indices: None
    
    Output：
        patient-level dataframe unique patient_id, and add covariates1：
            - Patient_ID 
            - covariates1 (0/1)
    """
    df = df.copy()

    # 1. str -> num 
    df["Test_Results_clean"] = df["Test_Results"].astype(str)

    df["Test_Results_clean"] = pd.to_numeric(df["Test_Results_clean"], errors="coerce")

    # 2. Initialized 
    df[covariates1] = 0

    # 3. Clinical definition of covariates1 
    # test2 ≥ threshold2
    df.loc[
        (df["Test"] == test2_val) & (df["Test_Results_clean"] >= threshold2),
        covariates1
    ] = 1

    # test1 ≥ threshold1
    df.loc[
        (df["Test"] == test1_val) & (df["Test_Results_clean"] >= threshold1),
        covariates1
    ] = 1

    # 4. merged by patinets, meet one of the test, covariates1 = 1 
    df_patient = (
        df.groupby("Patient_ID")[covariates1]
        .max()
        .reset_index()
    )

    return df_patient


def build_final_patient_df(df):
    """
    Merges the calculated covariates1 back into the main cohort.
    
    Uses indices: None
    
    Output:
        1 row per patient with a binary 'covariates1' indicator.
    """
    
    df = df.copy()

    df_covariates1= build_covariates1_patient_df(df)

    df_base = df.drop(columns=["Test", "Test_Results"], errors = 'ignore')
    df_base = df_base.drop_duplicates(subset=["Patient_ID"])

    df_final = df_base.merge(df_covariates1, on="Patient_ID", how="left")
    df_final[covariates1] = df_final[covariates1].fillna(0)

    return df_final

    
# Construct family history data
def process_family_history(family_df: str, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts family diagnoses history details.
    
    Input:
        family_df, final_cohort
        
    Uses indices from family:
        [3] fam_id (Patient_ID)
        [6] fam_text (DiagnosisText_orig)
        [10] fam_code (DiagnosisCode_orig)
        [12] fam_rel (Relationship_orig)

    Output:
        - FamilyRelationship (semicolon-separated list of family members)
        - FamilyDiagnosis (semicolon-separated list of family ICD-9 diagnosis codes)
        - FamilyDescription (semicolon-separated list of family disease descriptions)
        - Family_History (1 if family history of event exists, else 0)
    """
    # Check if empty OR if the dataset is missing the required indices
    required_max_index = max(fam_id, fam_text, fam_code, fam_rel)
    if family_df is None or family_df.empty or family_df.shape[1] <= required_max_index: 
        return final_cohort.assign(FamilyRelationship=np.nan, FamilyDiagnosis=np.nan, 
                                   FamilyDescription=np.nan, **{covariates2: 0})

    fam = family_df.copy()
    
    col_id = fam.columns[fam_id]
    col_rel = fam.columns[fam_rel]
    col_code = fam.columns[fam_code]
    col_text = fam.columns[fam_text]
    
    fam[col_id] = fam[col_id].astype(str)
    
    keep_cols = [col_id, col_rel, col_code, col_text]
    fam = fam[keep_cols].drop_duplicates()
    
    # Add the event indicator BEFORE grouping
    fam[covariates2] = fam[col_code].astype(str).str.contains(
        "|".join([str(c) for c in event_code]), na=False
    ).astype(int)

    def join_str(x):
        return "; ".join(x.dropna().unique().astype(str))

    fam_grouped = fam.groupby(col_id).agg(
        FamilyRelationship=(col_rel, join_str),
        FamilyDiagnosis=(col_code, join_str),
        FamilyDescription=(col_text, join_str),
        Family_History=(covariates2, "max")
    ).reset_index()
    
    fam_grouped.rename(columns={fam_grouped.columns[0]: "Patient_ID"}, inplace=True)

    expected_cols = ["FamilyRelationship", "FamilyDiagnosis", "FamilyDescription", covariates2]
    for c in expected_cols:
        if c in final_cohort.columns:
            final_cohort = final_cohort.drop(columns=[c])

    merged_cohort = final_cohort.merge(fam_grouped, on="Patient_ID", how="left")
    merged_cohort[covariates2] = merged_cohort[covariates2].fillna(0).astype(int)
    
    return merged_cohort


# Construct number of comorbidities data
def calculate_comorbidities(encounter_dx: pd.DataFrame, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the count of unique baseline diagnoses prior to the patient's index date.
    Excludes diagnoses related to the primary study conditions.
    
    Input:
        encounter_dx, final_cohort
        
    Uses indices from encounter_dx:
        [3] enc_id (Patient_iD)
        [11] enc_dx_code (DiagnosisCode_calc)
        [12] enc_date (DateCreated)
        [14] enc_icd3_num (ICD3_num)

    Output:
        - Num_Comorbidities (count of unique baseline diagnosis categories prior to the index_date)
    """
    # Check if empty OR if the dataset is missing the required indices
    required_max_index = max(enc_id, enc_dx_code, enc_date, enc_icd3_num)
    if encounter_dx is None or encounter_dx.empty or encounter_dx.shape[1] <= required_max_index:
        return final_cohort.assign(**{covariates3: 0})

    col_id = encounter_dx.columns[enc_id]
    col_date = encounter_dx.columns[enc_date]
    col_code = encounter_dx.columns[enc_dx_code]
    col_icd3 = encounter_dx.columns[enc_icd3_num]

    enc = encounter_dx[[col_id, col_date, col_code, col_icd3]].copy()
    enc[col_id] = enc[col_id].astype(str)
    enc[col_date] = pd.to_datetime(enc[col_date], errors="coerce")
    enc[col_icd3] = pd.to_numeric(enc[col_icd3], errors="coerce")
 
    # Merge with final cohort to get the patient's specific index_date
    merged = enc.merge(
        final_cohort[["Patient_ID", "index_date"]], 
        left_on=col_id, 
        right_on="Patient_ID",
        how="inner")
    
    # Filter for baseline (strictly BEFORE index date)
    baseline_dx = merged[merged[col_date] < merged["index_date"]].copy()
    
    # Exclude the primary study diseases from the baseline comorbidity count
    baseline_dx = baseline_dx[~baseline_dx[col_icd3].isin(study_codes)]
    
    # Count unique ICD3 disease categories per patient
    comorbid_counts = baseline_dx.groupby("Patient_ID")[col_code].nunique().reset_index()
    comorbid_counts.rename(columns={col_code: covariates3}, inplace=True)

    # Merge back and force type to integer (Handling NaNs as 0)
    merged_final = final_cohort.merge(comorbid_counts, on="Patient_ID", how="left")
    merged_final[covariates3] = merged_final[covariates3].fillna(0).astype(int)
    
    return merged_final
    
# Construct final table（exposure + event + duration）
def build_final_dataset(exposure1_patient: pd.DataFrame, 
                        exposure2_patient: pd.DataFrame, 
                        exposure3_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates the three separate exposure groups into a single dataset and calculates survival data.
    Input:
        exposure1_patient, exposure2_patient, exposure3_patient
        
    Uses indices: None

    Output:
        combined dataset with:
            - exposure (exposure1 / exposure2 / exposure3)
            - event (1 if exist, else 0)
            - end_date (follow-up end)
            - duration (time from index_date to event or censor)
    """
    # --- Step 1: copy ---
    exposure1_df = exposure1_patient.copy()
    exposure2_df = exposure2_patient.copy()
    exposure3_df = exposure3_patient.copy()

    # --- Step 2: add exposure ---
    exposure1_df["exposure"] = exposure1
    exposure2_df["exposure"] = exposure2
    exposure3_df["exposure"] = exposure3

    final_df = pd.concat([exposure1_df, exposure2_df, exposure3_df], axis=0, ignore_index=True)

    # --- Step 3: event and duration---
    final_df["event"] = final_df[event_date].notna().astype(int)
    final_df[event_date] = pd.to_datetime(final_df[event_date], errors="coerce")
    final_df["index_date"] = pd.to_datetime(final_df["index_date"], errors="coerce")
    
    final_df["end_date"] = final_df[event_date].fillna(censor_date)
    final_df["duration"] = (final_df["end_date"] - final_df["index_date"]).dt.days

    return final_df

def filter_target_population_pipeline():
    """
    Output:
        The final target population dataframe ready for analysis.
    """

   # --- Load encounter_dx ---
    encounter_dx = load_data(
        encounter_dx_path, 
        quoting=3, 
        on_bad_lines="skip"
    )

    # --- Load patient ---
    patient = load_data(patient_path)

    # --- Load labs ---
    labs_df = load_data(lab_path)

    # --- Load family history ---
    family_df = load_data(family_path)

    # --- Run Pipeline Steps
    encounter_dx_condition = add_condition_column(encounter_dx)
    
    patient_dx = build_patient_dx_dict(encounter_dx_condition)
    patient_with_dx = merge_patient_dx(patient, patient_dx)

    exposure1_patient, exposure2_patient, exposure3_patient = build_exposure_groups(patient_with_dx)

    final_patient = build_final_dataset(exposure1_patient, exposure2_patient, exposure3_patient)
    
    final_patient = calculate_comorbidities(encounter_dx, final_patient)
    encounter_dx_condition.drop(columns=['ICD3', 'ICD3_num'], inplace=True, errors='ignore')
    final_patient = process_lab_data(labs_df, final_patient)
    final_patient = build_final_patient_df(final_patient)
    final_patient = process_family_history(family_df, final_patient)
    
    final_patient.drop(columns = ['BirthMonth', 'OptedOut', 'OptOutDate'], inplace=True, errors='ignore')

    # --- Reorder columns ---
    front_cols = [
        "Patient_ID",
        "age",
        patient.columns[pat_sex],
        patient.columns[pat_birth],
        "dx_dict",
        "exposure",
        "index_date"
    ]

    back_cols = [covariates1, covariates2, covariates3, "event", "end_date", "duration"]

    all_cols = final_patient.columns.tolist()

    mid_cols = [c for c in all_cols if c not in front_cols and c not in back_cols]

    ordered_cols = (
        [c for c in front_cols if c in all_cols] +
        mid_cols +
        [c for c in back_cols if c in all_cols]
    )

    final_patient = final_patient[ordered_cols]

    return final_patient
