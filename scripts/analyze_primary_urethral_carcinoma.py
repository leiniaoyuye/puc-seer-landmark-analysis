"""Reproducible descriptive and conditional competing-risk analysis for SEER PUC.

Input is the user-exported, unmodified SEER*Stat Case Listing file and its
dictionary. The script creates aggregate outputs only; it does not export a
patient-level derivative file.

Study population:
  * Primary urethral malignancy selected in SEER*Stat (C68.0, 2004--2023)
  * Adults (>=18 years)
  * Not death-certificate-only reported
  * Microscopically confirmed
  * Epithelial carcinoma (ICD-O-3 morphology 8010--8576)
  * Known survival days and vital status
  * Earliest eligible urethral primary retained for each patient

Primary outcome:
  Event 1 = death attributable to urethral cancer (SEER cause-specific field)
  Event 2 = death attributable to another cause (SEER other-cause field)
  Event 0 = alive/censored

Landmarks: diagnosis, 1, 3 and 5 years. The analysis conditions on survival
strictly beyond 0, 365, 1095 and 1826 days, respectively. Aalen--Johansen CIF
estimates are reported over the following 1-, 3- and 5-year windows.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


RAW_DIR = Path(r"D:\文章写作\SEER\PUC")
RAW_TXT = RAW_DIR / "PUC_SEER17_Nov2025_2004_2023_RAW.txt"
RAW_DIC = RAW_DIR / "PUC_SEER17_Nov2025_2004_2023_RAW.dic"
OUT_DIR = Path(__file__).resolve().parent

MICROSCOPIC_CONFIRMATION = {
    "Positive histology",
    "Positive exfoliative cytology, no positive histology",
    "Positive microscopic confirm, method not specified",
}


def get_dictionary_columns(dic_path: Path) -> list[str]:
    """Read displayed column names from a SEER*Stat .dic file."""
    columns: list[str] = []
    for line in dic_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"Var\d+Name=(.*)", line)
        if match:
            columns.append(match.group(1))
    if len(columns) != 33:
        raise ValueError(f"Expected 33 exported fields, found {len(columns)}.")
    return columns


def read_raw_data() -> pd.DataFrame:
    columns = get_dictionary_columns(RAW_DIC)
    data = pd.read_csv(
        RAW_TXT,
        sep="\t",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
    )
    if data.shape[1] != len(columns):
        raise ValueError("Raw TXT column count does not match the SEER dictionary.")
    return data


def parse_age(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value.str.extract(r"(\d+)", expand=False), errors="coerce")


def derive_analysis_variables(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["age_years"] = parse_age(data["Age recode with single ages and 90+"])
    data["histology_code"] = pd.to_numeric(data["Histologic Type ICD-O-3"], errors="coerce")
    data["survival_days"] = pd.to_numeric(data["Survival Days"], errors="coerce")

    data["age_group"] = pd.cut(
        data["age_years"],
        bins=[17, 59, 69, 79, np.inf],
        labels=["18-59", "60-69", "70-79", "80+"],
    ).astype("string")
    data["diagnosis_period"] = np.where(
        pd.to_numeric(data["Year of diagnosis"], errors="coerce") <= 2013,
        "2004-2013",
        "2014-2023",
    )

    broad_histology = data["Histology recode - broad groupings"]
    data["histology_group"] = np.select(
        [
            broad_histology.eq("8120-8139: transitional cell papillomas and carcinomas"),
            broad_histology.eq("8050-8089: squamous cell neoplasms"),
            broad_histology.eq("8140-8389: adenomas and adenocarcinomas"),
            broad_histology.eq("8010-8049: epithelial neoplasms, NOS"),
        ],
        ["Urothelial carcinoma", "Squamous cell carcinoma", "Adenocarcinoma", "Epithelial NOS"],
        default="Other epithelial carcinoma",
    )

    stage = data["Combined Summary Stage with Expanded Regional Codes (2004+)"]
    data["stage_group"] = stage.replace(
        {
            "Localized only": "Localized",
            "Regional by direct extension only": "Regional: direct extension",
            "Regional lymph nodes involved only": "Regional: nodes only",
            "Regional by both direct extension and lymph node involvement": "Regional: extension + nodes",
            "Distant site(s)/node(s) involved": "Distant",
            "Unknown/unstaged/unspecified/DCO": "Unknown",
        }
    )
    data["stage_collapsed"] = np.select(
        [
            data["stage_group"].eq("Localized"),
            data["stage_group"].str.startswith("Regional", na=False),
            data["stage_group"].eq("Distant"),
        ],
        ["Localized", "Regional", "Distant"],
        default="Unknown",
    )

    cancer_death = data["SEER cause-specific death classification"].eq(
        "Dead (attributable to this cancer dx)"
    )
    other_death = data["SEER other cause of death classification"].eq(
        "Dead (attributable to causes other than this cancer dx)"
    )
    data["event"] = np.select([cancer_death, other_death], [1, 2], default=0).astype(int)
    data["event_label"] = data["event"].map(
        {0: "Alive/censored", 1: "Urethral cancer death", 2: "Other-cause death"}
    )
    unknown_cause_death = (
        data["Vital status recode (study cutoff used)"].eq("Dead")
        & data["SEER cause-specific death classification"].eq("Dead (missing/unknown COD)")
    )
    data.loc[unknown_cause_death, "event_label"] = "Death with unknown cause/censored"
    return data


def build_cohort(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the prespecified inclusion criteria and preserve a flow table."""
    rows: list[dict[str, object]] = []
    current = data.copy()
    rows.append({"step": "Raw malignant C68.0 urethra records exported from SEER*Stat", "n": len(current)})

    current = current.loc[current["age_years"] >= 18].copy()
    rows.append({"step": "Age >=18 years", "n": len(current)})

    current = current.loc[
        ~current["Type of Reporting Source"].eq("Death certificate only")
    ].copy()
    rows.append({"step": "Excluding death-certificate-only cases", "n": len(current)})

    current = current.loc[current["Diagnostic Confirmation"].isin(MICROSCOPIC_CONFIRMATION)].copy()
    rows.append({"step": "Microscopically confirmed", "n": len(current)})

    current = current.loc[current["histology_code"].between(8010, 8576)].copy()
    rows.append({"step": "Epithelial carcinoma (ICD-O-3 8010-8576)", "n": len(current)})

    current = current.loc[
        current["survival_days"].notna()
        & current["Vital status recode (study cutoff used)"].isin(["Alive", "Dead"])
    ].copy()
    rows.append({"step": "Known survival time and vital status", "n": len(current)})

    # SEER is tumor based. Retain one index record per person so that the
    # analysis unit is the patient and observations are independent. When a
    # person has more than one eligible urethral primary, use the earliest
    # diagnosis and then the lowest SEER record number as a deterministic
    # tie-breaker for diagnoses recorded in the same calendar year.
    current["diagnosis_year_numeric"] = pd.to_numeric(
        current["Year of diagnosis"], errors="coerce"
    )
    current["record_number_numeric"] = pd.to_numeric(
        current["Record number recode"], errors="coerce"
    )
    current = (
        current.sort_values(
            ["Patient ID", "diagnosis_year_numeric", "record_number_numeric"],
            kind="stable",
        )
        .drop_duplicates(subset="Patient ID", keep="first")
        .copy()
    )
    rows.append({"step": "Earliest eligible urethral primary per patient: final analytic cohort", "n": len(current)})
    return current, pd.DataFrame(rows)


def count_percent(series: pd.Series, denominator: int) -> pd.Series:
    count = series.value_counts(dropna=False, sort=False)
    return count.map(lambda x: f"{int(x)} ({100 * x / denominator:.1f})")


def make_baseline_table(cohort: pd.DataFrame) -> pd.DataFrame:
    variables: list[tuple[str, str, list[str] | None]] = [
        ("Sex", "Sex", ["Male", "Female"]),
        ("Age at diagnosis", "age_group", ["18-59", "60-69", "70-79", "80+"]),
        (
            "Race/ethnicity",
            "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
            [
                "Non-Hispanic White",
                "Non-Hispanic Black",
                "Hispanic (All Races)",
                "Non-Hispanic Asian or Pacific Islander",
                "Non-Hispanic American Indian/Alaska Native",
                "Non-Hispanic Unknown Race",
            ],
        ),
        ("Diagnosis period", "diagnosis_period", ["2004-2013", "2014-2023"]),
        (
            "Histology",
            "histology_group",
            [
                "Urothelial carcinoma",
                "Squamous cell carcinoma",
                "Adenocarcinoma",
                "Epithelial NOS",
                "Other epithelial carcinoma",
            ],
        ),
        (
            "Summary stage",
            "stage_group",
            [
                "Localized",
                "Regional: direct extension",
                "Regional: nodes only",
                "Regional: extension + nodes",
                "Distant",
                "Unknown",
            ],
        ),
        (
            "Marital status",
            "Marital status at diagnosis",
            [
                "Married (including common law)",
                "Single (never married)",
                "Widowed",
                "Divorced",
                "Separated",
                "Unmarried or Domestic Partner",
                "Unknown",
            ],
        ),
    ]
    output: list[dict[str, str]] = []
    n = len(cohort)
    for label, column, order in variables:
        frequencies = cohort[column].value_counts(dropna=False)
        categories = order or list(frequencies.index)
        for category in categories:
            count = int(frequencies.get(category, 0))
            output.append(
                {
                    "Variable": label,
                    "Category": category,
                    "Overall, n (%)": f"{count} ({100 * count / n:.1f})",
                    "n": count,
                    "percent": round(100 * count / n, 2),
                }
            )
    return pd.DataFrame(output)


def aalen_johansen(time: np.ndarray, event: np.ndarray, horizon: int) -> dict[str, float | int]:
    """Two-event Aalen--Johansen cumulative-incidence estimate at `horizon`."""
    keep = (time >= 0) & np.isfinite(time)
    time = time[keep]
    event = event[keep]
    n = len(time)
    survival = 1.0
    cif_cancer = 0.0
    cif_other = 0.0

    # Only event times alter the Aalen--Johansen estimator. Censoring times
    # remain in the risk set calculation through the sorted full follow-up time.
    event_mask = (event > 0) & (time <= horizon)
    event_times = time[event_mask]
    if len(event_times):
        unique_times, inverse = np.unique(event_times, return_inverse=True)
        at_risk = n - np.searchsorted(np.sort(time), unique_times, side="left")
        observed_events = event[event_mask]
        d_cancer = np.bincount(
            inverse, weights=(observed_events == 1), minlength=len(unique_times)
        )
        d_other = np.bincount(
            inverse, weights=(observed_events == 2), minlength=len(unique_times)
        )
        survival_before = np.cumprod(
            np.r_[1.0, 1.0 - (d_cancer + d_other) / at_risk]
        )[:-1]
        cif_cancer = float(np.sum(survival_before * d_cancer / at_risk))
        cif_other = float(np.sum(survival_before * d_other / at_risk))
        survival = float(np.prod(1.0 - (d_cancer + d_other) / at_risk))

    return {
        "n": n,
        "cif_urethral_cancer_death": cif_cancer,
        "cif_other_cause_death": cif_other,
        "event_free_survival": survival,
    }


def bootstrap_cif_ci(
    time: np.ndarray,
    event: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    repetitions: int = 1000,
) -> dict[str, float]:
    """Nonparametric percentile 95% CIs for the two CIFs at one horizon."""
    n = len(time)
    cancer = np.empty(repetitions)
    other = np.empty(repetitions)
    for rep in range(repetitions):
        sample = rng.integers(0, n, n)
        estimate = aalen_johansen(time[sample], event[sample], horizon)
        cancer[rep] = float(estimate["cif_urethral_cancer_death"])
        other[rep] = float(estimate["cif_other_cause_death"])
    return {
        "cif_urethral_cancer_death_ci95_lower_percent": round(100 * float(np.quantile(cancer, 0.025)), 2),
        "cif_urethral_cancer_death_ci95_upper_percent": round(100 * float(np.quantile(cancer, 0.975)), 2),
        "cif_other_cause_death_ci95_lower_percent": round(100 * float(np.quantile(other, 0.025)), 2),
        "cif_other_cause_death_ci95_upper_percent": round(100 * float(np.quantile(other, 0.975)), 2),
    }


def landmark_outputs(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    landmarks = {"Diagnosis": 0, "1-year survivor": 365, "3-year survivor": 1095, "5-year survivor": 1826}
    windows = {"1 year": 365, "3 years": 1095, "5 years": 1826}
    audit_rows: list[dict[str, object]] = []
    cif_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260827)

    for landmark_label, landmark_days in landmarks.items():
        # At diagnosis, all patients are eligible. Subsequent landmarks require survival past that day.
        landmark_cohort = cohort if landmark_days == 0 else cohort.loc[cohort["survival_days"] > landmark_days].copy()
        time_after_landmark = landmark_cohort["survival_days"].to_numpy(dtype=float) - landmark_days
        event = landmark_cohort["event"].to_numpy(dtype=int)

        for window_label, window_days in windows.items():
            cancer_events = int(np.sum((time_after_landmark <= window_days) & (event == 1)))
            other_events = int(np.sum((time_after_landmark <= window_days) & (event == 2)))
            alive_or_censored_before_window = int(
                np.sum((time_after_landmark < window_days) & (event == 0))
            )
            observed_to_window_or_event = int(
                np.sum((time_after_landmark >= window_days) | (event != 0))
            )
            audit_rows.append(
                {
                    "landmark": landmark_label,
                    "landmark_days": landmark_days,
                    "window": window_label,
                    "window_days": window_days,
                    "n_at_landmark": len(landmark_cohort),
                    "urethral_cancer_deaths_within_window": cancer_events,
                    "other_cause_deaths_within_window": other_events,
                    "alive_censored_before_window": alive_or_censored_before_window,
                    "observed_to_window_or_event": observed_to_window_or_event,
                }
            )
            estimate = aalen_johansen(time_after_landmark, event, window_days)
            ci = bootstrap_cif_ci(time_after_landmark, event, window_days, rng)
            cif_rows.append(
                {
                    "landmark": landmark_label,
                    "landmark_days": landmark_days,
                    "window": window_label,
                    "window_days": window_days,
                    **estimate,
                    "cif_urethral_cancer_death_percent": round(
                        100 * float(estimate["cif_urethral_cancer_death"]), 2
                    ),
                    "cif_other_cause_death_percent": round(
                        100 * float(estimate["cif_other_cause_death"]), 2
                    ),
                    "event_free_survival_percent": round(100 * float(estimate["event_free_survival"]), 2),
                    **ci,
                }
            )
    return pd.DataFrame(audit_rows), pd.DataFrame(cif_rows)


def landmark_stratified_cif(cohort: pd.DataFrame) -> pd.DataFrame:
    """Three-year conditional CIFs in prespecified clinical subgroups.

    This is a descriptive effect-modification screen, not a treatment-effect
    model. It is intentionally restricted to clinically interpretable groups
    and a three-year window, the study's prespecified primary conditional
    window.
    """
    landmarks = {"Diagnosis": 0, "1-year survivor": 365, "3-year survivor": 1095, "5-year survivor": 1826}
    strata = [
        ("Sex", "Sex", ["Male", "Female"]),
        ("Age at diagnosis", "age_group", ["18-59", "60-69", "70-79", "80+"]),
        ("Stage", "stage_collapsed", ["Localized", "Regional", "Distant", "Unknown"]),
        (
            "Histology",
            "histology_group",
            [
                "Urothelial carcinoma",
                "Squamous cell carcinoma",
                "Adenocarcinoma",
                "Epithelial NOS",
                "Other epithelial carcinoma",
            ],
        ),
    ]
    rows: list[dict[str, object]] = []
    horizon = 1095
    for landmark_label, landmark_days in landmarks.items():
        at_landmark = cohort if landmark_days == 0 else cohort.loc[cohort["survival_days"] > landmark_days].copy()
        for stratum_label, column, levels in strata:
            for level in levels:
                group = at_landmark.loc[at_landmark[column].eq(level)].copy()
                if group.empty:
                    continue
                time = group["survival_days"].to_numpy(dtype=float) - landmark_days
                event = group["event"].to_numpy(dtype=int)
                estimate = aalen_johansen(time, event, horizon)
                rows.append(
                    {
                        "landmark": landmark_label,
                        "landmark_days": landmark_days,
                        "stratifier": stratum_label,
                        "group": level,
                        "window": "3 years",
                        "window_days": horizon,
                        "n_at_landmark": len(group),
                        "urethral_cancer_deaths_within_window": int(np.sum((time <= horizon) & (event == 1))),
                        "other_cause_deaths_within_window": int(np.sum((time <= horizon) & (event == 2))),
                        "cif_urethral_cancer_death_percent": round(
                            100 * float(estimate["cif_urethral_cancer_death"]), 2
                        ),
                        "cif_other_cause_death_percent": round(
                            100 * float(estimate["cif_other_cause_death"]), 2
                        ),
                    }
                )
    return pd.DataFrame(rows)


def landmark_cause_specific_cox(cohort: pd.DataFrame) -> pd.DataFrame:
    """Parsimonious adjusted cause-specific Cox models in a 3-year window.

    These models are supplementary to the CIF analysis. Competing deaths are
    treated as censored for the cause being modelled. To preserve events per
    parameter at the 5-year landmark, adjustment is limited to age, sex and
    summary stage (all recorded at diagnosis).
    """
    landmarks = {"Diagnosis": 0, "3-year survivor": 1095, "5-year survivor": 1826}
    horizon = 1095
    rows: list[dict[str, object]] = []
    labels = {
        "age_per_10_years": "Age at diagnosis, per 10 years",
        "male": "Male (vs female)",
        "stage_regional": "Regional stage (vs localized)",
        "stage_distant": "Distant stage (vs localized)",
        "stage_unknown": "Unknown stage (vs localized)",
    }
    for landmark_label, landmark_days in landmarks.items():
        sample = cohort if landmark_days == 0 else cohort.loc[cohort["survival_days"] > landmark_days].copy()
        sample = sample.copy()
        elapsed = sample["survival_days"].to_numpy(dtype=float) - landmark_days
        model_time = np.minimum(elapsed, horizon)
        exog = pd.DataFrame(
            {
                "age_per_10_years": sample["age_years"].to_numpy(dtype=float) / 10.0,
                "male": sample["Sex"].eq("Male").astype(int).to_numpy(),
                "stage_regional": sample["stage_collapsed"].eq("Regional").astype(int).to_numpy(),
                "stage_distant": sample["stage_collapsed"].eq("Distant").astype(int).to_numpy(),
                "stage_unknown": sample["stage_collapsed"].eq("Unknown").astype(int).to_numpy(),
            }
        )
        for cause, cause_label in [(1, "Urethral cancer death"), (2, "Other-cause death")]:
            status = ((sample["event"].to_numpy(dtype=int) == cause) & (elapsed <= horizon)).astype(int)
            result = PHReg(model_time, exog, status=status).fit(disp=0)
            confidence = result.conf_int()
            for index, covariate in enumerate(exog.columns):
                rows.append(
                    {
                        "landmark": landmark_label,
                        "landmark_days": landmark_days,
                        "window": "3 years",
                        "n": len(sample),
                        "events_for_modelled_cause": int(status.sum()),
                        "outcome": cause_label,
                        "covariate": labels[covariate],
                        "hazard_ratio": round(float(np.exp(result.params[index])), 3),
                        "ci95_lower": round(float(np.exp(confidence[index, 0])), 3),
                        "ci95_upper": round(float(np.exp(confidence[index, 1])), 3),
                        "p_value": round(float(result.pvalues[index]), 5),
                    }
                )
    return pd.DataFrame(rows)


def unknown_cause_sensitivity(cohort: pd.DataFrame) -> pd.DataFrame:
    """Extreme-case sensitivity analysis for deaths with unknown cause."""
    landmarks = {"Diagnosis": 0, "5-year survivor": 1826}
    scenarios = {
        "Censored at death (primary analysis)": 0,
        "All assigned to urethral cancer death": 1,
        "All assigned to other-cause death": 2,
    }
    unknown = cohort["event_label"].eq("Death with unknown cause/censored").to_numpy()
    rows: list[dict[str, object]] = []
    for scenario, assigned_event in scenarios.items():
        event = cohort["event"].to_numpy(dtype=int).copy()
        event[unknown] = assigned_event
        for landmark, landmark_days in landmarks.items():
            eligible = (
                np.ones(len(cohort), dtype=bool)
                if landmark_days == 0
                else cohort["survival_days"].to_numpy(dtype=float) > landmark_days
            )
            estimate = aalen_johansen(
                cohort["survival_days"].to_numpy(dtype=float)[eligible] - landmark_days,
                event[eligible],
                1095,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "landmark": landmark,
                    "n_at_landmark": int(eligible.sum()),
                    "unknown_cause_deaths_in_landmark_cohort": int(unknown[eligible].sum()),
                    "cif_urethral_cancer_death_percent": round(
                        100 * float(estimate["cif_urethral_cancer_death"]), 2
                    ),
                    "cif_other_cause_death_percent": round(
                        100 * float(estimate["cif_other_cause_death"]), 2
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = derive_analysis_variables(read_raw_data())
    cohort, flow = build_cohort(raw)
    baseline = make_baseline_table(cohort)
    audit, cif = landmark_outputs(cohort)
    stratified_cif = landmark_stratified_cif(cohort)
    cox = landmark_cause_specific_cox(cohort)
    unknown_cod = unknown_cause_sensitivity(cohort)
    outcome_summary = pd.DataFrame(
        [
            {"follow_up_status": "Urethral cancer death", "n": int((cohort["event"] == 1).sum())},
            {"follow_up_status": "Other-cause death", "n": int((cohort["event"] == 2).sum())},
            {
                "follow_up_status": "Alive at study cutoff",
                "n": int(cohort["Vital status recode (study cutoff used)"].eq("Alive").sum()),
            },
            {
                "follow_up_status": "Death with unknown cause/censored",
                "n": int(cohort["event_label"].eq("Death with unknown cause/censored").sum()),
            },
        ]
    )

    flow.to_csv(OUT_DIR / "cohort_flow.csv", index=False, encoding="utf-8-sig")
    outcome_summary.to_csv(OUT_DIR / "outcome_status.csv", index=False, encoding="utf-8-sig")
    baseline.to_csv(OUT_DIR / "table1_baseline.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT_DIR / "landmark_event_audit.csv", index=False, encoding="utf-8-sig")
    cif.to_csv(OUT_DIR / "landmark_cif_estimates.csv", index=False, encoding="utf-8-sig")
    stratified_cif.to_csv(
        OUT_DIR / "landmark_stratified_cif_3years.csv", index=False, encoding="utf-8-sig"
    )
    cox.to_csv(OUT_DIR / "landmark_cause_specific_cox_3years.csv", index=False, encoding="utf-8-sig")
    unknown_cod.to_csv(OUT_DIR / "unknown_cause_death_sensitivity.csv", index=False, encoding="utf-8-sig")

    summary = [
        "SEER primary urethral carcinoma conditional competing-risk analysis",
        f"Final analytic cohort: {len(cohort):,}",
        f"Urethral cancer deaths: {int((cohort['event'] == 1).sum()):,}",
        f"Other-cause deaths: {int((cohort['event'] == 2).sum()):,}",
        f"Alive at study cutoff: {int(cohort['Vital status recode (study cutoff used)'].eq('Alive').sum()):,}",
        f"Deaths with unknown cause, censored at death: {int(cohort['event_label'].eq('Death with unknown cause/censored').sum()):,}",
        "",
        "CIF 95% confidence intervals use 1,000 nonparametric bootstrap samples.",
        "Outputs are aggregate tables. See the script docstring for cohort and outcome definitions.",
    ]
    (OUT_DIR / "README.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
