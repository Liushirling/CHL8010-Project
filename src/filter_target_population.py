import pandas as pd
import numpy as np
import os

BASE_PATH = "../raw/output"
encounter_dx_path = os.path.join(BASE_PATH, "encounter_dx_cleaned.csv")
patient_path = os.path.join(BASE_PATH, "patient_cleaned.csv")
lab_path: str = "../raw/output/lab_cleaned.csv"
family_path: str = "../raw/output/family_cleaned.csv"
censor_date = pd.to_datetime('2015-07-21')


def clean_column_names(df):
    df.columns = df.columns.str.replace('"', '', regex=False)
    return df

def add_condition_column(df: pd.DataFrame) -> pd.DataFrame:

    df["DiagnosisCodeType_calc"] = (
        df["DiagnosisCodeType_calc"]
        .astype(str)
        .str.upper()
    )

    df["ICD3"] = (
        df["DiagnosisCode_calc"]
        .astype(str)
        .str.split(".")
        .str[0]
    )

    df["condition"] = pd.Series(dtype='object')

    mask_icd9 = df["DiagnosisCodeType_calc"] == "ICD9"

    # depression
    depression_codes = {"296", "300", "309", "311"}
    df.loc[
        mask_icd9 & df["ICD3"].isin(depression_codes),
        "condition"
    ] = "depression"

    # asthma
    df.loc[
        mask_icd9 &
        df["condition"].isna() &
        (df["ICD3"] == "493"),
        "condition"
    ] = "asthma"

    # copd
    # convert ICD3 to numeric safely
    df["ICD3_num"] = pd.to_numeric(df["ICD3"], errors="coerce")

    df.loc[
        mask_icd9 &
        df["condition"].isna() &
        df["ICD3_num"].between(490, 496),
        "condition"
    ] = "copd"

    df.drop(columns=["ICD3", "ICD3_num"], inplace=True)

    return df

def build_patient_dx_dict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output:
        Patient_ID | dx_dict
        dx_dict: {condition: [datetime]}
    """

    df_valid = df[df["condition"].notna()].copy()

    df_valid["DateCreated"] = pd.to_datetime(
        df_valid["DateCreated"],
        errors="coerce"
    )

    df_valid = df_valid[df_valid["DateCreated"].notna()]

    def build_dict(group):
        dx_dict = {}

        for _, row in group.iterrows():
            cond = row["condition"]
            date = row["DateCreated"]  

            if cond not in dx_dict:
                dx_dict[cond] = []

            dx_dict[cond].append(date)

        for k in dx_dict:
            dx_dict[k] = sorted(dx_dict[k])

        return dx_dict

    result = (
        df_valid
        .groupby("Patient_ID")
        .apply(build_dict)
        .reset_index(name="dx_dict")
    )

    return result

def merge_patient_dx(patient_df: pd.DataFrame, 
                     patient_dx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dx_dict into patient table.

    Output:
        patient_df + dx_dict column
    """

    patient_df["Patient_ID"] = patient_df["Patient_ID"].astype(str)
    patient_dx_df["Patient_ID"] = patient_dx_df["Patient_ID"].astype(str)

    merged = patient_df.merge(
        patient_dx_df,
        on="Patient_ID",
        how="inner"   
    )

    return merged

# construct cohort
def build_exposure_groups(patient_with_dx: pd.DataFrame):
    """
    Input:
        patient_with_dx
        
    Output::
        - first_asthma (date of first asthma diagnosis)
        - second_asthma (date of second asthma diagnosis)
        - first_copd (date of first COPD diagnosis)
        - depression_date (date of first lifetime depression event)
        - index_date (date of first asthma / copd / aco diagnosis)
        - age (patient's exact age calculated at the index_date)
    """
    
    df = patient_with_dx.copy()

    def get_nth_date(d, condition, n):
        if not isinstance(d, dict) or condition not in d:
            return pd.NaT
            
        dates = sorted(d[condition])
        
        return dates[n] if len(dates) > n else pd.NaT

    df["first_asthma"] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, "asthma", 0))
    
    df["second_asthma"] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, "asthma", 1))
    
    df["first_copd"] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, "copd", 0))
    
    df["depression_date"] = df["dx_dict"].apply(
        lambda d: get_nth_date(d, "depression", 0))


    # 1. ACO (Asthma-COPD overlap)
    aco_mask = (
        df["first_copd"].notna() &
        df["second_asthma"].notna() &
        (df["first_copd"] < df["first_asthma"]))
    aco_group = df[aco_mask].copy()
    aco_group["index_date"] = aco_group["second_asthma"] # ACO index date is 2nd asthma

    # exclude if depression occurs between initial COPD and index date (2nd asthma)
    aco_group = aco_group[
        aco_group["depression_date"].isna() |
        ~((aco_group["depression_date"] >= aco_group["first_copd"]) &
          (aco_group["depression_date"] <= aco_group["index_date"]))]

    # 2. COPD
    copd_mask = (
        df["first_copd"].notna() &
        (df["first_copd"] < df["first_asthma"].fillna(pd.Timestamp.max)) &
        ~aco_mask)
    copd_group = df[copd_mask].copy()
    copd_group["index_date"] = copd_group["first_copd"]

    # exclude if depression occurs on/before index date
    copd_group = copd_group[
        copd_group["depression_date"].isna() |
        (copd_group["depression_date"] > copd_group["index_date"])]

    # 3. Asthma
    asthma_mask = (
        df["first_asthma"].notna() &
        (df["first_asthma"] <= df["first_copd"].fillna(pd.Timestamp.max)) &
        ~aco_mask)
    asthma_group = df[asthma_mask].copy()
    asthma_group["index_date"] = asthma_group["first_asthma"]

    # exclude if depression occurs on/before index date
    asthma_group = asthma_group[
        asthma_group["depression_date"].isna() |
        (asthma_group["depression_date"] > asthma_group["index_date"])]

    # calculate age based on the index dates
    def calc_age(group_df):
        group_df["BirthYear"] = pd.to_numeric(group_df["BirthYear"], errors = "coerce")
        group_df["age"] = group_df["index_date"].dt.year - group_df["BirthYear"]
        return group_df

    return calc_age(asthma_group), calc_age(copd_group), calc_age(aco_group)

# construct lab data
def process_lab_data(labs_df: str, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        labs_df, final_cohort
        
    Output:
        - Test (Glucose, HbA1c, or Other)
        - Test_Results (lab result strictly prior to index_date, "Unknown" if no baseline lab exists)
    """
    
    if labs_df is None or labs_df.empty:
        return final_cohort

    labs = labs_df.copy()

    labs.columns = labs.columns.str.strip().str.replace('"', '', regex=False)
    labs["Patient_ID"] = labs["Patient_ID"].astype(str)

    merged_labs = labs.merge(final_cohort[["Patient_ID", "index_date"]],
                             on="Patient_ID", how="inner")
    
    merged_labs["ObservationDate"] = pd.to_datetime(merged_labs["DateCreated"], errors="coerce")

    baseline_labs = merged_labs[merged_labs["ObservationDate"] <= merged_labs["index_date"]].copy()
    
    numeric_results = pd.to_numeric(baseline_labs['TestResult_orig'], errors='coerce')
    baseline_labs["Test_Results"] = numeric_results.fillna('Unknown')

    substring_to_standard = {
        'glucose': 'Glucose',
        'hba1c':   'HbA1c'
    }

    def standardize_name(name):
        if pd.isna(name) or str(name).strip() == "nan" or str(name).strip() == "": 
            return np.nan
            
        name_lower = str(name).lower()
        for sub, std in substring_to_standard.items():
            if sub in name_lower:
                return std
        return "Other"

    baseline_labs["Test"] = baseline_labs["Name_orig"].apply(standardize_name)
    
    baseline_labs = baseline_labs.dropna(subset=["Test"])

    baseline_labs = baseline_labs.sort_values(by=["Patient_ID", "Test", "ObservationDate"])
    recent_labs = baseline_labs.drop_duplicates(subset=["Patient_ID", "Test"], keep="last")

    long_format_labs = recent_labs[["Patient_ID", "Test", "Test_Results"]]

    merged_final = final_cohort.merge(long_format_labs, on="Patient_ID", how="left")

    merged_final["Test_Results"] = merged_final["Test_Results"].fillna("Unknown")
    
    return merged_final

# Construct family history data
def process_family_history(family_df: str, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        family_df, final_cohort

    Output:
        - FamilyRelationship (semicolon-separated list of family members)
        - FamilyDiagnosis (semicolon-separated list of family ICD-9 diagnosis codes)
        - FamilyDescription (semicolon-separated list of family disease descriptions)
    """
    
    if family_df is None or family_df.empty: 
        return final_cohort

    family = family_df.copy()
    
    family.columns = family.columns.str.strip().str.replace('"', '', regex=False)
    family["Patient_ID"] = family["Patient_ID"].astype(str)

    cols = ["Patient_ID", "Relationship_orig", "DiagnosisCode_orig", "DiagnosisText_orig"]
    family = family[cols].drop_duplicates()

    family.rename(columns={
        "Relationship_orig": "FamilyRelationship",
        "DiagnosisCode_orig": "FamilyDiagnosis",
        "DiagnosisText_orig": "FamilyDescription"}, inplace = True)

    
    fam_grouped = family.groupby("Patient_ID").agg(
        lambda x: "; ".join(x.dropna().unique())
    ).reset_index()
    
    return final_cohort.merge(fam_grouped, on="Patient_ID", how="left")

# Construct number of comorbidities data
def calculate_comorbidities(encounter_dx: pd.DataFrame, final_cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        encounter_dx, final_cohort

    Output:
        - Num_Comorbidities (count of unique baseline diagnosis categories prior to the index_date)
    """
    
    enc = encounter_dx[["Patient_ID", "DateCreated", "DiagnosisCode_calc"]].copy()
    enc["Patient_ID"] = enc["Patient_ID"].astype(str)
    enc["DateCreated"] = pd.to_datetime(enc["DateCreated"], errors="coerce")
 
    # Merge with final cohort to get the patient's specific index_date
    merged = enc.merge(final_cohort[["Patient_ID", "index_date"]], on="Patient_ID", how="inner")
    
    # Filter for baseline (strictly BEFORE index date)
    baseline_dx = merged[merged["DateCreated"] < merged["index_date"]].copy()
    baseline_dx["ICD3"] = baseline_dx["DiagnosisCode_calc"].astype(str).str.split(".").str[0]
    
    # Exclude the primary study diseases from the baseline comorbidity count
    # (Asthma: 493 | COPD: 490-496 | Depression: 296, 300, 309, 311)
    study_icd3_codes = {"490", "491", "492", "493", "494", "495", "496", "296", "300", "309", "311"}
    baseline_dx = baseline_dx[~baseline_dx["ICD3"].isin(study_icd3_codes)]
    
    # Count unique ICD3 disease categories per patient
    comorbid_counts = baseline_dx.groupby("Patient_ID")["DiagnosisCode_calc"].nunique().reset_index()
    comorbid_counts.rename(columns={"DiagnosisCode_calc": "Num_Comorbidities"}, inplace=True)

    # Merge back and force type to integer (Handling NaNs as 0)
    merged_final = final_cohort.merge(comorbid_counts, on="Patient_ID", how="left")
    merged_final["Num_Comorbidities"] = merged_final["Num_Comorbidities"].fillna(0).astype(int)
    
    return merged_final
    
# Construct final table（exposure + event + duration）
def build_final_dataset(asthma_patient: pd.DataFrame, copd_patient: pd.DataFrame, aco_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        asthma_patient, copd_patient, aco_patient

    Output:
        combined dataset with:
            - exposure (asthma / copd / aco)
            - event (1 if depression, else 0)
            - end_date (follow-up end)
            - duration (time from index_date to event or censor)
    """
    # --- Step 1: copy ---
    asthma_df = asthma_patient.copy()
    copd_df = copd_patient.copy()
    aco_df = aco_patient.copy()

    # --- Step 2: add exposure ---
    asthma_df["exposure"] = "asthma"
    copd_df["exposure"] = "copd"
    aco_df["exposure"] = "aco"

    final_df = pd.concat([asthma_df, copd_df, aco_df], axis=0, ignore_index=True)

    # --- Step 3: event and duration---
    final_df["event"] = final_df["depression_date"].notna().astype(int)
    final_df["depression_date"] = pd.to_datetime(final_df["depression_date"], errors="coerce")
    final_df["index_date"] = pd.to_datetime(final_df["index_date"], errors="coerce")
    
    final_df["end_date"] = final_df["depression_date"].fillna(censor_date)
    final_df["duration"] = (final_df["end_date"] - final_df["index_date"]).dt.days

    return final_df

def filter_target_population_pipeline():

    # --- Load encounter_dx ---
    encounter_dx_pickle = encounter_dx_path.replace(".csv", ".pkl")
    if os.path.exists(encounter_dx_pickle):
        encounter_dx = pd.read_pickle(encounter_dx_pickle)
    else:
        encounter_dx = pd.read_csv(encounter_dx_path, dtype=str, sep="|", engine="python", quoting=3, on_bad_lines="skip")
        encounter_dx.to_pickle(encounter_dx_pickle)
        
    # --- Load patient ---
    patient_pickle = patient_path.replace(".csv", ".pkl")
    if os.path.exists(patient_pickle):
        patient = pd.read_pickle(patient_pickle)
    else:
        patient = pd.read_csv(patient_path, dtype=str, sep="|", engine="python")
        patient.to_pickle(patient_pickle)

    # --- Load labs ---
    lab_pickle = lab_path.replace(".csv", ".pkl")
    if os.path.exists(lab_pickle):
        labs_df = pd.read_pickle(lab_pickle)
    elif os.path.exists(lab_path):
        labs_df = pd.read_csv(lab_path, dtype=str, sep="|", engine="python")
        labs_df.to_pickle(lab_pickle)
    else:
        labs_df = pd.DataFrame()

    # --- Load family history ---
    family_pickle = family_path.replace(".csv", ".pkl")
    if os.path.exists(family_pickle):
        family_df = pd.read_pickle(family_pickle)
    elif os.path.exists(family_path):
        family_df = pd.read_csv(family_path, dtype=str, sep="|", engine="python")
        family_df.to_pickle(family_pickle)
    else:
        family_df = pd.DataFrame()

    encounter_dx_clean = clean_column_names(encounter_dx)
    encounter_dx_condition = add_condition_column(encounter_dx_clean)

    patient_dx = build_patient_dx_dict(encounter_dx_condition)
    patient_with_dx = merge_patient_dx(patient, patient_dx)

    asthma_patient, copd_patient, aco_patient = build_exposure_groups(patient_with_dx)

    final_patient = build_final_dataset(asthma_patient, copd_patient, aco_patient)
    
    final_patient = calculate_comorbidities(encounter_dx_clean, final_patient)
    final_patient = process_lab_data(labs_df, final_patient)
    final_patient = process_family_history(family_df, final_patient)

    return final_patient

 