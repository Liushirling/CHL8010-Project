
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# importfunctions
from data_input import read_csv_with_badlines, get_output_dir, clean_encounter_dx_bad_text, fix_encounter_dx_still_bad_lines, merge_good_cleaned_and_still_bad, filter_bad_cleaned_keep_only_readable, fix_lab_bad_lines

from data_cleaning import (
    clean_dataset
)

from filter_target_population import filter_target_population_pipeline

# survival analysis
from survival_model import run_survival_analysis




# Constant
patient_path: str = "../raw/C4MPatient.csv"
encounter_dx_path: str = "../raw/C4MEncounterdiagnosis.csv"
lab_path: str = "../raw/C4MLab.csv"
family_path: str = "../raw/C4MFamilyHistory.csv"
sep: str = "|"
chunksize: int = 100_000
output_dir: str = "../raw/output"
censor_date = pd.to_datetime('2015-07-21')

out = get_output_dir(output_dir)

files = {
    "encounter_dx": "encounter_dx_final.csv",
    "lab": "lab_final.csv",
    "family": "family_final.csv",
    "patient": "patient_final.csv"
}


# Input data
# Patient
patient, patient_bad = read_csv_with_badlines(patient_path, sep, chunksize)
patient.to_csv(out / "patient_final.csv", sep=sep, index=False)
patient.to_pickle(out / "patient_final.pkl")
print("Output patient_final.csv")

# Encounter diagnosis
encounter_dx, encounter_dx_bad = read_csv_with_badlines(encounter_dx_path, sep, chunksize)
print(f"[stage1] Encounter_dx good rows: {len(encounter_dx)} | bad rows: {len(encounter_dx_bad)}")
encounter_dx.to_csv(out / "encounter_dx_final.csv", sep=sep, index=False)
encounter_dx_bad.to_csv(out / "encounter_dx_bad.csv", index=False)
clean_encounter_dx_bad_text(str(out / "encounter_dx_bad.csv"), str(out / "encounter_dx_bad_cleaned.csv"))
filter_bad_cleaned_keep_only_readable(
    good_path=str(out / "encounter_dx_final.csv"),
    bad_cleaned_path=str(out / "encounter_dx_bad_cleaned.csv"),
    out_cleaned_path=str(out / "encounter_dx_bad_cleaned_filtered.csv"),
    out_still_bad_path=str(out / "encounter_dx_bad_still_unreadable.csv"),
    sep=sep,
    skip_first_line=False,
    label="encounter_dx",
)
out_path, n = fix_encounter_dx_still_bad_lines(
    input_path="../raw/output/encounter_dx_bad_still_unreadable.csv",
    output_path="../raw/output/encounter_dx_bad_still_unreadable_fixed.csv",
    expected_n_cols=13,
)
merge_good_cleaned_and_still_bad(
    good_path="../raw/output/encounter_dx_final.csv",
    bad_cleaned_path="../raw/output/encounter_dx_bad_cleaned_filtered.csv",
    still_bad_fixed_path="../raw/output/encounter_dx_bad_still_unreadable_fixed.csv",
    out_path="../raw/output/encounter_dx_final.csv",
    sep="|",
    label="Encounter_dx"
)

encounter_dx.to_pickle(out / "encounter_dx_final.pkl")
print("Output encounter_dx_final.csv")

# Lab
lab,_ = read_csv_with_badlines(lab_path, sep, chunksize)
lab.to_csv(out / "lab_final.csv", sep=sep, index=False)
lab.to_pickle(out / "lab_final.pkl")
print("Output lab_final.csv")

#Family history
family, family_bad = read_csv_with_badlines(family_path, sep, chunksize)
print(f"[stage1] Family good rows: {len(family)} | bad rows: {len(family_bad)}")
family.to_csv(out / "family_final.csv", sep=sep, index=False)
family.to_pickle(out / "family_final.pkl")

print("Output family_final.csv")

# DATA CLEANING 
clean_dataset(files["encounter_dx"], "encounter_dx_cleaned.csv")
clean_dataset(files["lab"], "lab_cleaned.csv")
clean_dataset(files["family"], "family_cleaned.csv")
clean_dataset(files["patient"], "patient_cleaned.csv")

print("\nAll datasets cleaned successfully.")

#filter_target_population and feature engineering
final_patient = filter_target_population_pipeline()
print("final_patient shape:", final_patient.shape)
print(final_patient.head())

print("duration NA:", final_patient["duration"].isna().sum())
print("duration length:", len(final_patient["duration"]))

print("event distribution:")
print(final_patient["event"].value_counts())

print("exposure distribution:")
print(final_patient["exposure"].value_counts())
# survival analysis
run_survival_analysis(final_patient)

print("\nAll computations completed.")
