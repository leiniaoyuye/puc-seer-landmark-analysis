"""Run formal Schoenfeld-residual tests for all cause-specific Cox models."""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_primary_urethral_carcinoma as analysis


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "cox_ph_assumption_tests.csv"
R_SCRIPT = ROOT / "check_cox_ph_assumptions.R"
R_EXECUTABLE = Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe")


def build_diagnostic_input(cohort: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    landmarks = {"Diagnosis": 0, "3-year survivor": 1095, "5-year survivor": 1826}
    outcomes = {1: "Urethral cancer death", 2: "Other-cause death"}
    horizon = 1095

    for landmark, landmark_days in landmarks.items():
        sample = (
            cohort.copy()
            if landmark_days == 0
            else cohort.loc[cohort["survival_days"] > landmark_days].copy()
        )
        elapsed = sample["survival_days"].to_numpy(dtype=float) - landmark_days
        follow_up_time = np.minimum(elapsed, horizon)
        for event_code, outcome in outcomes.items():
            rows.append(
                pd.DataFrame(
                    {
                        "landmark": landmark,
                        "outcome": outcome,
                        "follow_up_time": follow_up_time,
                        "status": (
                            (sample["event"].to_numpy(dtype=int) == event_code)
                            & (elapsed <= horizon)
                        ).astype(int),
                        "age_per_10_years": sample["age_years"].to_numpy(dtype=float) / 10.0,
                        "male": sample["Sex"].eq("Male").astype(int).to_numpy(),
                        "stage": sample["stage_collapsed"].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cohort, _ = analysis.build_cohort(
        analysis.derive_analysis_variables(analysis.read_raw_data())
    )
    diagnostic_input = build_diagnostic_input(cohort)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # R on this Windows installation cannot reliably receive Unicode paths on
    # the command line, so use an ASCII-only temporary directory and copy the
    # small R program there for execution. Only the aggregate test result is
    # retained after the temporary patient-level diagnostic input is deleted.
    with tempfile.TemporaryDirectory(prefix="seer_puc_ph_", dir=r"D:\SEER") as temporary_directory:
        input_path = Path(temporary_directory) / "ph_input.csv"
        temporary_r_script = Path(temporary_directory) / "check_ph.R"
        temporary_output = Path(temporary_directory) / "ph_results.csv"
        shutil.copy2(R_SCRIPT, temporary_r_script)
        diagnostic_input.to_csv(input_path, index=False, encoding="utf-8")
        subprocess.run(
            [str(R_EXECUTABLE), str(temporary_r_script), str(input_path), str(temporary_output)],
            check=True,
        )
        result = pd.read_csv(temporary_output)
        result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    if len(result) != 24:
        raise ValueError(f"Expected 24 PH-test rows, found {len(result)}")


if __name__ == "__main__":
    main()
