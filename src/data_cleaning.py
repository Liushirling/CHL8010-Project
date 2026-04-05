import pandas as pd
import os

BASE_PATH = "/Users/kieran/Desktop/UT/biosta/Winter/CHL 8010/8010Github/CHL8010-Project/data/output"

files = {
    "encounter_dx": "encounter_dx_final.csv",
    "lab": "lab_final.csv",
    "family": "family_final.csv",
    "patient": "patient_final.csv"
}


# strip quotes from header
def strip_quotes_from_header(df):
    df = df.copy()
    df.columns = (
        df.columns     
          .str.strip('"')   
    )
    return df

#### clean data (duplication check function)
def check_duplicate_by_first_column(df, df_name):
    """
    Check duplicates based on the first column (assumed primary key).
    According to schema: df.columns[0] is the table-specific ID.
    """
    pk_col = df.columns[0]

    print(f"\nChecking primary key: {pk_col}")
    print(f"Total rows: {len(df)}")

    dup = df[df[pk_col].duplicated(keep=False)]

    if dup.empty:
        print(f"No duplicate primary keys found. ({df_name})")
    else:
        print(f"Found {dup.shape[0]} duplicated rows ({df_name}):")
        print(dup.sort_values(pk_col))


# replace na and pseudo na to pd.NA
def standardize_missing_values(df):
    pseudo_missing = ["","Unknown", "UNK", "N/A", "NA","NAN", "None"]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .str.strip()
                .replace(pseudo_missing, pd.NA)
            )
    return df

# Categorical columns standardization
def standardize_categorical_columns(df, categorical_cols):

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.upper()
            )

    return df

# Date variable standardization
def convert_date_columns(df):
    """
    Find columns whose name contains 'date' and convert them to datetime.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if "date" in col.lower():
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

    return df_clean

# Check invalid dates
def report_invalid_dates(df,df_name):
    """
    year < 1900
    year > current year
    """

    current_year = pd.Timestamp.today().year

    for col in df.columns:
        if "date" in col.lower():
            dt = pd.to_datetime(df[col], errors="coerce")

            invalid = dt[
                (dt.dt.year < 1900) |
                (dt.dt.year > current_year)
            ]

            if not invalid.empty:
                print(f"Invalid dates found in column({df_name}): {col}")
                print(invalid.value_counts())


# convert_numeric_and_remove_negative
def convert_numeric_cols_and_remove_negative(df, cols):
    df_clean = df.copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean.loc[df_clean[col] < 0, col] = pd.NA

    return df_clean

# clean header
def clean_column_names(df):
    df.columns = df.columns.str.replace('"', '', regex=False)
    return df

def standardize_diagnosis_code_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DiagnosisCodeType_calc is uppercase string.
    """
    if "DiagnosisCodeType_calc" in df.columns:
        df["DiagnosisCodeType_calc"] = (
            df["DiagnosisCodeType_calc"]
            .astype(str)
            .str.upper()
        )
    return df


def create_icd3_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ICD3 column ONLY for ICD9 rows.
    Example: 493.90 -> 493
    """
    if "DiagnosisCode_calc" in df.columns and "DiagnosisCodeType_calc" in df.columns:
        
        # initialize column
        df["ICD3"] = pd.NA

        # mask for ICD9
        mask_icd9 = df["DiagnosisCodeType_calc"].astype(str).str.upper() == "ICD9"

        df.loc[mask_icd9, "ICD3"] = (
            df.loc[mask_icd9, "DiagnosisCode_calc"]
            .astype(str)
            .str.split(".")
            .str[0]
        )

    return df


def convert_icd3_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely convert ICD3 to numeric.
    Non-numeric values become NaN.
    """
    if "ICD3" in df.columns:
        df["ICD3_num"] = pd.to_numeric(df["ICD3"], errors="coerce")
    return df


# Output cleaned dataset
def output_df(df, output_path):
    #bad line
    df.to_csv(output_path, sep = "|", index=False)


# ========= CLEAN FUNCTION =========
def clean_dataset(file_name, output_name, num_cols=[]):
    input_path = os.path.join(BASE_PATH, file_name)
    output_path = os.path.join(BASE_PATH, output_name)

    print(f"\nCleaning: {file_name}")

    # --- Try loading pickle first (faster) ---
    pickle_path = input_path.replace(".csv", ".pkl")

    if os.path.exists(pickle_path):
        print(f"Loading from pickle: {pickle_path}")
        df = pd.read_pickle(pickle_path)
    else:
        print("Pickle not found, reading CSV...")
        df = pd.read_csv(
            input_path,
            dtype=str,
            sep="|",
            engine="python",
            quoting=3,
            on_bad_lines="skip"
        )

    # 1. strip header quotes
    df = strip_quotes_from_header(df)

    # 2. clean header
    df = clean_column_names(df)

    # 3. standardize missing
    df = standardize_missing_values(df)

    # 4. convert dates
    df = convert_date_columns(df)

    # 5. convert numeric columns
    df = convert_numeric_cols_and_remove_negative(df, num_cols)

    # 6. ICD-related preprocessing (only if columns exist)
    if "DiagnosisCode_calc" in df.columns and "DiagnosisCodeType_calc" in df.columns:
        df = standardize_diagnosis_code_type(df)
        df = create_icd3_column(df)
        df = convert_icd3_to_numeric(df)

    # 7. duplicate check (based on schema: first column = primary key)
    check_duplicate_by_first_column(df, file_name)

    # 8. output
    output_df(df, output_path)

    print(f"Saved cleaned file to: {output_path}")