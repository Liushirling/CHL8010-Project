import pandas as pd
import numpy as np


BASE_PATH = "/Users/kieran/Desktop/UT/biosta/Winter/CHL 8010/8010Github/CHL8010-Project/data/output"
patient_path: str = "../data/C4MPatient.csv"
encounter_dx_path: str = "../data/C4MEncounterdiagnosis.csv"
lab_path: str = "../data/C4MLab.csv"
family_path: str = "../data/C4MFamilyHistory.csv"
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

    df["condition"] = np.nan

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

def merge_patient_dx(
    patient_df: pd.DataFrame,
    patient_dx_df: pd.DataFrame
) -> pd.DataFrame:
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

# compute index_date and age
def add_age_and_index(patient_with_dx: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
        - index_date: first asthma or copd date
        - age: index_date year - BirthYear
    """

    df = patient_with_dx.copy()

    df["dx_dict"] = df["dx_dict"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    def first_exposure_date(d):
        dates = []
        if "asthma" in d:
            dates.extend(d["asthma"])
        if "copd" in d:
            dates.extend(d["copd"])
        return min(dates) if len(dates) > 0 else pd.NaT

    df["index_date"] = df["dx_dict"].apply(first_exposure_date)

    df["index_date"] = pd.to_datetime(df["index_date"], errors="coerce")

    df["BirthYear"] = pd.to_numeric(df["BirthYear"], errors="coerce")

    df["age"] = df["index_date"].dt.year - df["BirthYear"]

    return df

# construct cohort
def build_exposure_groups(patient_with_dx_index_date_age: pd.DataFrame):


    df = patient_with_dx_index_date_age.copy()

    df["dx_dict"] = df["dx_dict"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    def has_asthma(d):
        return "asthma" in d

    def has_copd(d):
        return "copd" in d

    def has_depression(d):
        return "depression" in d

    def first_date(d, key):
        return min(d[key]) if key in d else pd.NaT

    asthma_group = df[
        df["dx_dict"].apply(has_asthma) &
        ~df["dx_dict"].apply(has_copd)
    ].copy()

    copd_group = df[
        df["dx_dict"].apply(has_copd) &
        ~df["dx_dict"].apply(has_asthma)
    ].copy()

    asthma_group["index_date"] = asthma_group["dx_dict"].apply(
        lambda d: first_date(d, "asthma")
    )

    copd_group["index_date"] = copd_group["dx_dict"].apply(
        lambda d: first_date(d, "copd")
    )

    asthma_group["depression_date"] = asthma_group["dx_dict"].apply(
        lambda d: first_date(d, "depression")
    )

    copd_group["depression_date"] = copd_group["dx_dict"].apply(
        lambda d: first_date(d, "depression")
    )

    asthma_group = asthma_group[
        asthma_group["depression_date"].isna() |
        (asthma_group["depression_date"] >= asthma_group["index_date"])
    ]

    copd_group = copd_group[
        copd_group["depression_date"].isna() |
        (copd_group["depression_date"] >= copd_group["index_date"])
    ]

    return asthma_group, copd_group


# Construct final table（exposure + event + duration）
def build_final_dataset(asthma_patient: pd.DataFrame, copd_patient: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        asthma_patient, copd_patient

    Output:
        combined dataset with:
            - exposure (asthma / copd)
            - event (1 if depression, else 0)
            - duration (time from index_date to event or censor)
    """

    # --- Step 1: copy ---
    asthma_df = asthma_patient.copy()
    copd_df = copd_patient.copy()

    # --- Step 2: add exposure ---
    asthma_df["exposure"] = "asthma"
    copd_df["exposure"] = "copd"

    # --- Step 3: event ---
    asthma_df["event"] = asthma_df["depression_date"].notna().astype(int)
    copd_df["event"] = copd_df["depression_date"].notna().astype(int)

    # --- Step 4: duration ---
    # asthma
    asthma_df["depression_date"] = pd.to_datetime(asthma_df["depression_date"], errors="coerce")
    asthma_df["index_date"] = pd.to_datetime(asthma_df["index_date"], errors="coerce")
    asthma_df["end_date"] = asthma_df["depression_date"].fillna(censor_date)
    asthma_df["duration"] = (asthma_df["end_date"] - asthma_df["index_date"]).dt.days

    # copd
    copd_df["depression_date"] = pd.to_datetime(copd_df["depression_date"], errors="coerce")
    copd_df["index_date"] = pd.to_datetime(copd_df["index_date"], errors="coerce")
    copd_df["end_date"] = copd_df["depression_date"].fillna(censor_date)
    copd_df["duration"] = (copd_df["end_date"] - copd_df["index_date"]).dt.days

    # merge
    final_df = pd.concat([asthma_df, copd_df], axis=0, ignore_index=True)

    return final_df





def filter_target_population_pipeline():
    encounter_dx = pd.read_csv(
        encounter_dx_path,
        dtype=str,
        sep="|",
        engine="python",
    )
    patient = pd.read_csv(
        patient_path,
        dtype=str,
        sep=",",
        engine="python",
    )

    encounter_dx_clean = clean_column_names(encounter_dx)
    encounter_dx_condition = add_condition_column(encounter_dx_clean)

    patient_dx = build_patient_dx_dict(encounter_dx_condition)
    patient_with_dx = merge_patient_dx(patient, patient_dx)

    patient_with_dx_index_date_age = add_age_and_index(patient_with_dx)

    asthma_patient, copd_patient = build_exposure_groups(
        patient_with_dx_index_date_age
    )

    final_patient = build_final_dataset(asthma_patient, copd_patient)

    return final_patient





