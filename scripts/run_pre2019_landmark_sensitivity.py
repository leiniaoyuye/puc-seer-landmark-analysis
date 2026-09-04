"""Assess calendar-era truncation in the 5-year PUC landmark analysis.

The primary cohort includes diagnoses through 2023. This sensitivity analysis
restricts the index diagnosis year to 2004-2018, so every participant had the
opportunity to reach the prespecified 5-year landmark before the 2023 cutoff.
Only aggregate results are written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze_primary_urethral_carcinoma import (
    build_cohort,
    derive_analysis_variables,
    landmark_outputs,
    read_raw_data,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "seer_puc_conditional_competing_risks"


def main() -> None:
    raw = derive_analysis_variables(read_raw_data())
    cohort, _ = build_cohort(raw)
    cohort = cohort.loc[cohort["diagnosis_year_numeric"] <= 2018].copy()

    audit, cif = landmark_outputs(cohort)
    five_year = cif.loc[cif["landmark"].eq("5-year survivor")].copy()
    five_year.insert(0, "analysis", "Diagnosis years 2004-2018")
    five_year.insert(1, "rationale", "Every participant could reach the 5-year landmark before the 2023 study cutoff")
    five_year.to_csv(
        OUT / "sensitivity_pre2019_5year_landmark_cif.csv",
        index=False,
        encoding="utf-8-sig",
    )

    five_year_audit = audit.loc[audit["landmark"].eq("5-year survivor")].copy()
    five_year_audit.insert(0, "analysis", "Diagnosis years 2004-2018")
    five_year_audit.to_csv(
        OUT / "sensitivity_pre2019_5year_landmark_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Restricted index cohort: {len(cohort):,}")
    print(five_year.to_string(index=False))


if __name__ == "__main__":
    main()
