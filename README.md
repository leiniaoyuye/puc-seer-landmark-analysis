# Conditional competing mortality after primary urethral carcinoma

Reproducible analysis materials for the manuscript **“Conditional Competing Mortality after Primary Urethral Carcinoma: A Population-Based SEER Landmark Analysis (2004–2023)”**.

## Contents

- `scripts/`: cohort construction, competing-risk estimation, Cox-model diagnostics, sensitivity analysis, and figure-generation scripts.
- `data/aggregate/`: aggregate tables used for manuscript tables and figures. No individual-level records are included.
- `figures/`: publication figures in PNG, TIFF, SVG, and PDF formats.

## Data access and privacy

The analysis uses the November 2025 SEER 17 Registries Research Data submission (2004–2023). Individual-level SEER records are not redistributed in this repository. Researchers wishing to reproduce the analysis should obtain access from the National Cancer Institute SEER Program, complete the SEER Data-Use Agreement, and place the authorized case-listing export and dictionary in a local, non-versioned data directory.

The original SEER export is intentionally excluded from version control. The `.sl` selection file and `.dic` field dictionary are provided only to document the field selection and cohort specification.

## Software environment

Python 3.10 or later is recommended. Install the Python dependencies with:

```text
pip install -r requirements.txt
```

The proportional-hazards diagnostic script requires R with the `survival` package. The scripts create aggregate outputs only; they do not write patient-level derivative files.

## Reproduction outline

1. Obtain authorized SEER data and place the raw `.txt` and `.dic` files in a local data directory.
2. Update the input paths in `scripts/analyze_primary_urethral_carcinoma.py` for the local environment.
3. Run the main analysis script to generate cohort flow, baseline, landmark CIF, subgroup, Cox, and unknown-cause sensitivity tables.
4. Run the figure-generation script using the generated aggregate tables.
5. Run the R/Python diagnostic scripts to reproduce the proportional-hazards checks and the pre-2019 landmark sensitivity analysis.

## Study definition

The analytic cohort consists of adults with microscopically confirmed epithelial primary urethral carcinoma (SEER site C68.0), diagnosed during 2004–2023, with known survival information. The earliest eligible urethral primary was retained per patient. Urethral cancer death and other-cause death were treated as competing events. Landmark cohorts were defined at diagnosis and after 365, 1095, and 1826 days of survival.

## License

Code and aggregate outputs are released under the MIT License. SEER data remain subject to the SEER Data-Use Agreement and are not covered by this repository license.


## Submission and reproducibility note

This repository supports the manuscript submitted to *Cancers*. The main entry points are `scripts/analyze_primary_urethral_carcinoma.py` for cohort derivation and aggregate statistical outputs, `scripts/make_seer_puc_figures.py` for the publication figures, and the R/Python diagnostic scripts for proportional-hazards and calendar-eligibility checks. The aggregate CSV files in `data/aggregate/` correspond to the tables and figures supplied with the manuscript. Because SEER individual-level records are governed by the SEER Data-Use Agreement, a researcher must obtain authorized access independently; the repository contains no patient-level SEER data. Last verified: 5 September 2026.
