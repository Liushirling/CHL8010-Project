
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
patient_path: str = "../data/C4MPatient.csv"
encounter_dx_path: str = "../data/C4MEncounterdiagnosis.csv"
lab_path: str = "../data/C4MLab.csv"
family_path: str = "../data/C4MFamilyHistory.csv"
sep: str = "|"
chunksize: int = 100_000
output_dir: str = "../data/output"
censor_date = pd.to_datetime('2015-07-21')

out = get_output_dir(output_dir)

files = {
    "encounter_dx": "encounter_dx_final.csv",
    "lab": "lab_final.csv",
    "family": "family_good.csv",
    "patient": "patient_good.csv"
}


# Input data
# Patient
patient, patient_bad = read_csv_with_badlines(patient_path, sep, chunksize)
patient.to_csv(out / "patient_good.csv", sep=sep, index=False)

print("Output patient_final.csv")

# Encounter diagnosis
encounter_dx, encounter_dx_bad = read_csv_with_badlines(encounter_dx_path, sep, chunksize)
print(f"[stage1] Encounter_dx good rows: {len(encounter_dx)} | bad rows: {len(encounter_dx_bad)}")
encounter_dx.to_csv(out / "encounter_dx_good.csv", sep=sep, index=False)
encounter_dx_bad.to_csv(out / "encounter_dx_bad.csv", index=False)
clean_encounter_dx_bad_text(str(out / "encounter_dx_bad.csv"), str(out / "encounter_dx_bad_cleaned.csv"))
filter_bad_cleaned_keep_only_readable(
    good_path=str(out / "encounter_dx_good.csv"),
    bad_cleaned_path=str(out / "encounter_dx_bad_cleaned.csv"),
    out_cleaned_path=str(out / "encounter_dx_bad_cleaned_filtered.csv"),
    out_still_bad_path=str(out / "encounter_dx_bad_still_unreadable.csv"),
    sep=sep,
    skip_first_line=False,
    label="encounter_dx",
)
out_path, n = fix_encounter_dx_still_bad_lines(
    input_path="../data/output/encounter_dx_bad_still_unreadable.csv",
    output_path="../data/output/encounter_dx_bad_still_unreadable_fixed.csv",
    expected_n_cols=13,
)
merge_good_cleaned_and_still_bad(
    good_path="../data/output/encounter_dx_good.csv",
    bad_cleaned_path="../data/output/encounter_dx_bad_cleaned_filtered.csv",
    still_bad_fixed_path="../data/output/encounter_dx_bad_still_unreadable_fixed.csv",
    out_path="../data/output/encounter_dx_final.csv",
    sep="|",
    label="Encounter_dx"
)

print("Output encounter_dx_final.csv")

# Lab
lab,_ = read_csv_with_badlines(lab_path, sep, chunksize)
lab.to_csv(out / "lab_final.csv", sep=sep, index=False)

print("Output lab_final.csv")

# # Family history
family, family_bad = read_csv_with_badlines(family_path, sep, chunksize)
print(f"[stage1] Family good rows: {len(family)} | bad rows: {len(family_bad)}")
family.to_csv(out / "family_good.csv", sep=sep, index=False)

print("Output family_final.csv")

# DATA CLEANING 
clean_dataset(files["encounter_dx"], "encounter_dx_cleaned.csv")
clean_dataset(files["lab"], "lab_cleaned.csv")
clean_dataset(files["family"], "family_cleaned.csv")
clean_dataset(files["patient"], "patient_cleaned.csv")

print("\nAll datasets cleaned successfully.")

#filter_target_population and feature engineering
final_patient = filter_target_population_pipeline()

# survival analysis
run_survival_analysis(final_patient)
