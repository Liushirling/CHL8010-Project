import pandas as pd
from typing import Tuple, List
from pandas.errors import ParserError
import pandas as pd
from typing import Tuple, List
import io
from typing import Optional
from pathlib import Path
pd.set_option("display.max_columns", None)

def _try_parse(lines, sep, col_names):
    return pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=sep,
        header=None,
        names=col_names,
        engine="python"
    )

def _find_first_bad_line(buffer, sep, col_names):
    lo, hi = 0, len(buffer)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            _try_parse(buffer[:mid], sep, col_names)
            lo = mid
        except ParserError:
            hi = mid
    idx = hi - 1
    return idx, buffer[idx]

def read_csv_with_badlines(filepath: str, sep: str = "|", chunksize: int = 100_000) -> Tuple[pd.DataFrame, pd.DataFrame]:

    good_chunks: List[pd.DataFrame] = []
    bad_lines: List[str] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
        col_names = header.split(sep)
        n_cols = len(col_names)
        buffer = []
        
        for line in f:
            line = line.rstrip("\r\n")  

            if line.count(sep) == n_cols - 1 and (line.count('"') % 2 == 0):
                buffer.append(line)
            else:
                bad_lines.append(line)

            if len(buffer) >= chunksize:
                while buffer:
                    try:
                        chunk_df = _try_parse(buffer, sep, col_names)
                        good_chunks.append(chunk_df)
                        buffer.clear()
                    except ParserError:
                        bad_idx, bad_line = _find_first_bad_line(buffer, sep, col_names)
                        bad_lines.append(bad_line)
                        buffer.pop(bad_idx) 

        if buffer:
            while buffer:
                try:
                    chunk_df = _try_parse(buffer, sep, col_names)
                    good_chunks.append(chunk_df)
                    buffer.clear()
                except ParserError:
                    bad_idx, bad_line = _find_first_bad_line(buffer, sep, col_names)
                    bad_lines.append(bad_line)
                    buffer.pop(bad_idx)

    df_good = pd.concat(good_chunks, ignore_index=True) if good_chunks else pd.DataFrame(columns=col_names)
    df_bad = pd.DataFrame({"raw_line": bad_lines})

    return df_good, df_bad

# Paths 
def get_output_dir(base_dir: str = "../raw/output") -> Path:
    out = Path(base_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out

# Handling bad-lines files for Encounter_diagnosis
def clean_encounter_dx_bad_text(
    input_path: str,
    output_path: str | None = None,
    encoding: str = "utf-8",
) -> Tuple[str, int]:
    """
    For Encounter DX bad-lines exported file (plain text processing, no pandas):
    - skip first header line 'raw_line'
    - remove all double quotes and backslashes 
    - if the last character is NOT '0', drop the last character
    - write cleaned lines to output

    Returns: (output_path, number_of_lines_written)
    """
    in_path = Path(input_path)
    if output_path is None:
        output_path = str(in_path.with_name(in_path.stem + "_cleaned.csv"))
    out_path = Path(output_path)

    n_written = 0
    with open(in_path, "r", encoding=encoding, errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8", newline="\n") as fout:

        # Skip header line: raw_line
        next(fin, None)

        for line in fin:
            s = line.rstrip("\r\n")
            if not s:
                continue

            # remove quotes and backslashes
            s = s.replace('"', "")
            s = s.replace("\\", "")

            # if last char isn't '0', drop it
            if s and s[-1] != "0":
                s = s[:-1]

            if not s:
                continue

            fout.write(s + "\n")
            n_written += 1

    print("[Encounter_dx] Wrote:", out_path)
    print("[Encounter_dx] Lines written:", n_written)
    return None

def filter_bad_cleaned_keep_only_readable(
    good_path: str,
    bad_cleaned_path: str,
    out_cleaned_path: str,
    sep: str = "|",
    out_still_bad_path: Optional[str] = None,
    skip_first_line: bool = False,
    encoding: str = "utf-8",
    label: str = "",
) -> None:
    """
    Stream through bad_cleaned (plain text), keep only lines that are readable
    with the schema defined by good_path, and write them to out_cleaned_path.

    Also optionally write dropped lines unchanged to out_still_bad_path.
    """
    prefix = f"[{label}] " if label else ""

    good_path = Path(good_path)
    bad_cleaned_path = Path(bad_cleaned_path)
    out_cleaned_path = Path(out_cleaned_path)

    if not good_path.exists():
        raise FileNotFoundError(f"{prefix}Good file not found: {good_path.resolve()}")
    if not bad_cleaned_path.exists():
        raise FileNotFoundError(f"{prefix}Bad cleaned file not found: {bad_cleaned_path.resolve()}")

    # Read only header from good to define schema
    with open(good_path, "r", encoding=encoding, errors="ignore") as f:
        header = f.readline().rstrip("\r\n")
    col_names = header.split(sep)
    n_cols = len(col_names)

    out_cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    still_bad_fout = None
    if out_still_bad_path is not None:
        out_still_bad_path = Path(out_still_bad_path)
        out_still_bad_path.parent.mkdir(parents=True, exist_ok=True)
        still_bad_fout = open(out_still_bad_path, "w", encoding="utf-8", newline="\n")

    n_kept = 0
    n_dropped = 0

    with open(bad_cleaned_path, "r", encoding=encoding, errors="ignore") as fin, \
         open(out_cleaned_path, "w", encoding="utf-8", newline="\n") as fout:

        if skip_first_line:
            next(fin, None)

        for raw in fin:
            line = raw.rstrip("\r\n")
            if not line:
                continue

            # 1) quick field-count check
            parts = line.split(sep)
            if len(parts) != n_cols:
                n_dropped += 1
                if still_bad_fout:
                    still_bad_fout.write(line + "\n")
                continue

            # 2) strict parse check
            try:
                _ = pd.read_csv(
                    io.StringIO(line),
                    sep=sep,
                    header=None,
                    names=col_names,
                    engine="python",
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception:
                n_dropped += 1
                if still_bad_fout:
                    still_bad_fout.write(line + "\n")
                continue

            fout.write(line + "\n")
            n_kept += 1

    if still_bad_fout:
        still_bad_fout.close()

    print(f"{prefix}Expected columns: {n_cols}")
    print(f"{prefix}Kept readable lines: {n_kept}")
    print(f"{prefix}Dropped still-bad lines: {n_dropped}")
    print(f"{prefix}Filtered bad_cleaned written to: {out_cleaned_path.resolve()}")
    if out_still_bad_path is not None:
        print(f"{prefix}Still-bad lines written to: {Path(out_still_bad_path).resolve()}")
    
    return None


# dealing with still-bad lines
def fix_encounter_dx_still_bad_lines(
    input_path: str,
    output_path: str,
    *,
    sep: str = "|",
    expected_n_cols: int = 13,
    encoding: str = "utf-8",
) -> Tuple[str, int]:
    """
    Fix still-unreadable encounter_dx bad lines by:
      - reading as plain text
      - if a line has MORE than expected_n_cols fields,
        truncate to the first expected_n_cols fields
      - write cleaned lines to output file

    This specifically fixes cases like:
    ...|2004-02-09|2001000000034213   <-- extra trailing field

    Returns
    -------
    (output_path, n_lines_written)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    with open(input_path, "r", encoding=encoding, errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8", newline="\n") as fout:

        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split(sep)

            if len(parts) > expected_n_cols:
                parts = parts[:expected_n_cols]

            fout.write(sep.join(parts) + "\n")
            n_written += 1

    return str(output_path), n_written


# lab bad line function
def fix_lab_bad_lines(
    input_path: str,
    output_path: str,
    *,
    sep: str = "|",
    expected_n_cols: int = 21,
    encoding: str = "utf-8",
) -> tuple[str, int]:
    """
    Fix lab bad lines with embedded '|' inside text fields.

    Strategy:
    - remove quotes and escape chars
    - split by '|'
    - keep first N_head columns
    - keep last N_tail columns
    - merge middle part into ONE column
    """

    from pathlib import Path

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    N_HEAD = 13
    N_TAIL = 1   

    with open(input_path, "r", encoding=encoding, errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8", newline="\n") as fout:

        # skip header if exists
        first = next(fin, None)
        if first and "raw_line" not in first:
            fin = [first] + list(fin)

        for line in fin:
            line = line.strip()
            if not line:
                continue

            line = line.replace('"', '')
            line = line.replace('\\', '')

            parts = line.split(sep)

            if len(parts) == expected_n_cols:
                fout.write(sep.join(parts) + "\n")
                n_written += 1
                continue

            if len(parts) > expected_n_cols:

                head = parts[:N_HEAD]
                tail = parts[-N_TAIL:]

                middle = parts[N_HEAD:-N_TAIL]

                middle_joined = " ".join(middle)

                new_parts = head + [middle_joined] + tail

                if len(new_parts) < expected_n_cols:
                    new_parts += [""] * (expected_n_cols - len(new_parts))

                if len(new_parts) > expected_n_cols:
                    new_parts = new_parts[:expected_n_cols]

                fout.write(sep.join(new_parts) + "\n")
                n_written += 1

            else:
                parts += [""] * (expected_n_cols - len(parts))
                fout.write(sep.join(parts) + "\n")
                n_written += 1

    print(f"[Lab] Fixed lines written: {n_written}")
    return str(output_path), n_written


# Simple lab cleaner and merger
def simple_clean_and_merge_lab(
    bad_raw_path: str,
    lab_good_path: str,
    lab_final_path: str,
    *,
    sep: str = "|",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Very simple lab fixer:
      - remove ALL double quotes (")
      - remove ALL backslashes (\)
      - write cleaned file
      - read cleaned file using lab_good schema
      - merge with lab_good
      - write final merged file

    Returns merged DataFrame.
    """

    bad_raw_path = Path(bad_raw_path)
    lab_good_path = Path(lab_good_path)
    lab_final_path = Path(lab_final_path)

    if not bad_raw_path.exists():
        raise FileNotFoundError(f"Bad raw file not found: {bad_raw_path.resolve()}")
    if not lab_good_path.exists():
        raise FileNotFoundError(f"Lab good file not found: {lab_good_path.resolve()}")

    # 1) Clean raw file (remove " and \)
    cleaned_temp_path = bad_raw_path.with_name(bad_raw_path.stem + "_simple_cleaned.csv")

    with open(bad_raw_path, "r", encoding=encoding, errors="ignore") as fin, \
         open(cleaned_temp_path, "w", encoding="utf-8", newline="\n") as fout:

        # skip header raw_line if present
        first = next(fin, None)
        if first and "raw_line" not in first:
            fout.write(first.replace('"', "").replace("\\", ""))

        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue
            line = line.replace('"', "")
            line = line.replace("\\", "")
            fout.write(line + "\n")

    print(f"[Lab] Simple cleaned file written to: {cleaned_temp_path.resolve()}")

    # 2) Read good (has header)
    lab_good = pd.read_csv(
        lab_good_path,
        sep=sep,
        dtype=str,
        keep_default_na=False
    )
    colnames = lab_good.columns.tolist()

    # 3) Read cleaned bad using good schema
    lab_bad_cleaned = pd.read_csv(
        cleaned_temp_path,
        sep=sep,
        header=None,
        names=colnames,
        dtype=str,
        keep_default_na=False
    )

    print(f"[Lab] Good shape: {lab_good.shape}")
    print(f"[Lab] Bad cleaned shape: {lab_bad_cleaned.shape}")

    # 4) Merge
    lab_final = pd.concat([lab_good, lab_bad_cleaned], ignore_index=True)

    # 5) Write final
    lab_final_path.parent.mkdir(parents=True, exist_ok=True)
    lab_final.to_csv(lab_final_path, sep=sep, index=False)

    print(f"[Lab] Final merged shape: {lab_final.shape}")
    print(f"[Lab] Lab final written to: {lab_final_path.resolve()}")

    return lab_final



# merge good + cleaned_bad_filtered + still_bad -> final
def merge_good_cleaned_and_still_bad(
    good_path: str,
    bad_cleaned_path: str,
    still_bad_fixed_path: str,
    out_path: str,
    sep: str = "|",
    label: str = "",
) -> pd.DataFrame:
    """
    Merge:
      1) good CSV (HAS header)
      2) cleaned bad CSV (NO header)
      3) still-bad-fixed CSV (NO header)
    into one dataset using the good file's schema.

    Returns merged DataFrame and writes to out_path.
    """
    prefix = f"[{label}] " if label else ""

    good_path = Path(good_path)
    bad_cleaned_path = Path(bad_cleaned_path)
    still_bad_fixed_path = Path(still_bad_fixed_path)
    out_path = Path(out_path)

    if not good_path.exists():
        raise FileNotFoundError(f"{prefix}Good file not found: {good_path.resolve()}")
    if not bad_cleaned_path.exists():
        raise FileNotFoundError(f"{prefix}Bad cleaned file not found: {bad_cleaned_path.resolve()}")
    if not still_bad_fixed_path.exists():
        raise FileNotFoundError(f"{prefix}Still-bad-fixed file not found: {still_bad_fixed_path.resolve()}")

    # 1) Read good (has header)
    good = pd.read_csv(
        good_path,
        sep=sep,
        dtype=str,
        keep_default_na=False
    )
    print(f"{prefix}Good shape: {good.shape} ({good_path.name})")

    colnames = good.columns.tolist()

    # 2) Read bad_cleaned (NO header), force schema from good
    bad_cleaned = pd.read_csv(
        bad_cleaned_path,
        sep=sep,
        header=None,
        names=colnames,
        dtype=str,
        keep_default_na=False
    )
    print(f"{prefix}Bad cleaned shape: {bad_cleaned.shape} ({bad_cleaned_path.name})")

    # 3) Read still_bad_fixed (NO header), force schema from good
    still_bad = pd.read_csv(
        still_bad_fixed_path,
        sep=sep,
        header=None,
        names=colnames,
        dtype=str,
        keep_default_na=False
    )
    print(f"{prefix}Still-bad-fixed shape: {still_bad.shape} ({still_bad_fixed_path.name})")

    # 4) Concat
    merged = pd.concat([good, bad_cleaned, still_bad], ignore_index=True)
    print(f"{prefix}Merged shape: {merged.shape}")

    # 5) Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep=sep, index=False)
    print(f"{prefix}Merged data written to: {out_path.resolve()}")

    # Safety check
    assert list(good.columns) == list(bad_cleaned.columns) == list(still_bad.columns), (
        f"{prefix}Schema mismatch among inputs"
    )

    return merged

