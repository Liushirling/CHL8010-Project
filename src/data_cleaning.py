import pandas as pd
import os

BASE_PATH = "../raw/output"

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
def check_duplicate_by_first_column(df,primary_col,df_name):
    pk_col = primary_col
    print(f"\nChecking primary key: {pk_col}")
    print(f"Total rows: {len(df)}")
    dup = df[df[pk_col].duplicated(keep=False)]

    if dup.empty:
        print(f"No duplicate primary keys found.({df_name})")
    else:
        print(f"Found {dup.shape[0]} duplicated rows({df_name}):")
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

# Output cleaned dataset
def output_df(df, output_path):
    #bad line
    df.to_csv(output_path, sep = "|", index=False)



# ========= CLEAN FUNCTION =========
def clean_dataset(file_name, output_name, num_cols = []):
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

    # 2. standardize missing
    df = standardize_missing_values(df)

    # 3. convert dates
    df = convert_date_columns(df)

    # 4. convert numeric columns (auto detect numeric-looking columns)
    df = convert_numeric_cols_and_remove_negative(df, num_cols)

    # 5. clean header
    df = clean_column_names(df)

    # 6. output
    output_df(df, output_path)

    print(f"Saved cleaned file to: {output_path}")

