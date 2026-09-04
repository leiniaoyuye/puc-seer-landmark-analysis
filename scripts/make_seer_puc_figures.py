"""Create publication-grade figures for the SEER primary urethral carcinoma study.

Backend: Python/matplotlib only.
All plotted values are read from aggregate analysis outputs produced by
``analyze_primary_urethral_carcinoma.py``. Exports include editable SVG/PDF,
600-dpi TIFF, and a 300-dpi PNG preview.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
FIG_DIR = ROOT

CANCER = "#B64342"
CANCER_LIGHT = "#F2D4D0"
OTHER = "#3775BA"
OTHER_LIGHT = "#D8E5F3"
INK = "#272727"
MID = "#767676"
PALE = "#F3F3F3"
ACCENT = "#D8B365"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.solid_capstyle": "round",
    }
)


def export_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", length=3, width=0.7, color=INK)


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
             facecolor: str, edgecolor: str = "#B7B7B7", bold_first: bool = False) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.9,
    )
    ax.add_patch(box)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + width / 2, y + height * 0.61, lines[0], ha="center", va="center",
                fontsize=9, fontweight="bold", color=INK)
        ax.text(x + width / 2, y + height * 0.34, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=7.2, color=INK)
    else:
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=7.4, color=INK, linespacing=1.25)


def make_figure1_flow() -> None:
    flow = pd.read_csv(DATA_DIR / "cohort_flow.csv")
    outcomes = pd.read_csv(DATA_DIR / "outcome_status.csv").set_index("follow_up_status")["n"]
    if flow["n"].tolist() != [2828, 2827, 2804, 2774, 2648, 2648, 2638]:
        raise ValueError("Cohort-flow counts differ from the audited values.")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.08, 0.81), 0.56, 0.115,
             "2,828 malignant primary urethral tumors\nSEER 17 Registries, 2004–2023",
             "#E8EEF6", edgecolor="#9DB4D1", bold_first=True)
    draw_box(ax, (0.08, 0.62), 0.56, 0.105,
             "2,827 adults aged ≥18 years", "#F7F7F7", bold_first=True)
    draw_box(ax, (0.08, 0.43), 0.56, 0.105,
             "2,804 non–death-certificate-only cases", "#F7F7F7", bold_first=True)
    draw_box(ax, (0.08, 0.24), 0.56, 0.105,
             "2,774 microscopically confirmed tumors", "#F7F7F7", bold_first=True)
    # Separate patient-level inclusion and follow-up boxes keep the analysis
    # unit and the treatment of unknown cause of death explicit.
    final_x, final_y, final_w, final_h = 0.08, 0.025, 0.56, 0.145
    final_box = FancyBboxPatch(
        (final_x, final_y), final_w, final_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#EAF3EA", edgecolor="#9CBC9C", linewidth=0.9,
    )
    ax.add_patch(final_box)
    ax.text(final_x + final_w / 2, final_y + final_h * 0.72,
            "2,638 patients",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(final_x + final_w / 2, final_y + final_h * 0.36,
            "Earliest of 2,648 eligible epithelial records retained",
            ha="center", va="center", fontsize=7.3, color=INK)
    outcome_x, outcome_w = 0.69, 0.25
    outcome_box = FancyBboxPatch(
        (outcome_x, final_y), outcome_w, final_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor="#F7F7F7", edgecolor="#BBBBBB", linewidth=0.8,
    )
    ax.add_patch(outcome_box)
    ax.text(outcome_x + outcome_w / 2, final_y + final_h * 0.80, "Follow-up status",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
    outcome_specs = [
        (f"Urethral cancer death  {outcomes['Urethral cancer death']:,}", CANCER),
        (f"Other-cause death  {outcomes['Other-cause death']:,}", OTHER),
        (f"Alive  {outcomes['Alive at study cutoff']:,}", "#A8A8A8"),
        (f"Unknown cause/censored  {outcomes['Death with unknown cause/censored']:,}", "#D2A64B"),
    ]
    for ypos, (label, color) in zip([0.099, 0.076, 0.053, 0.030], outcome_specs):
        ax.scatter([outcome_x + 0.028], [ypos], s=18, color=color, zorder=3)
        ax.text(outcome_x + 0.047, ypos, label, ha="left", va="center", fontsize=5.4, color=INK)

    for y1, y2 in [(0.81, 0.725), (0.62, 0.535), (0.43, 0.345), (0.24, 0.16)]:
        ax.add_patch(FancyArrowPatch((0.36, y1), (0.36, y2), arrowstyle="-|>",
                                     mutation_scale=10, lw=0.9, color=MID))

    exclusions = [
        (0.735, "1 patient aged <18 years"),
        (0.545, "23 death-certificate-only cases"),
        (0.355, "30 without microscopic confirmation"),
        (0.195, "126 non-epithelial morphologies\n10 later eligible records"),
    ]
    for y, text in exclusions:
        ax.add_patch(FancyArrowPatch((0.64, y + 0.022), (0.72, y + 0.022), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.8, color="#AAAAAA"))
        ax.text(0.74, y + 0.022, text, ha="left", va="center", fontsize=7, color=MID)

    fig.suptitle("Study cohort and outcome classification", x=0.51, y=0.995,
                 fontsize=10, fontweight="bold", color=INK)
    export_figure(fig, "figure1_cohort_flow")


def annotated_heatmap(ax: plt.Axes, matrix: np.ndarray, row_labels: list[str],
                      col_labels: list[str], vmax: float, title: str,
                      secondary: np.ndarray | None = None, mask_below_n: int | None = None) -> mpl.image.AxesImage:
    cmap = LinearSegmentedColormap.from_list("competing", [CANCER, "#FAFAFA", OTHER], N=256)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F0F0F0",
                                           edgecolor="#B0B0B0", hatch="////", linewidth=0.5))
                ax.text(j, i, "n<30", ha="center", va="center", fontsize=6.3, color=MID)
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.52 else INK
            if secondary is None:
                label = f"{value:+.1f}"
            else:
                label = f"{value:+.1f}\n(n={int(secondary[i, j]):,})"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.6,
                    color=text_color, linespacing=1.15, fontweight="bold" if secondary is None else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def make_figure2_competing_mortality() -> None:
    cif = pd.read_csv(DATA_DIR / "landmark_cif_estimates.csv")
    strat = pd.read_csv(DATA_DIR / "landmark_stratified_cif_3years.csv")
    cox = pd.read_csv(DATA_DIR / "landmark_cause_specific_cox_3years.csv")
    order = ["Diagnosis", "1-year survivor", "3-year survivor", "5-year survivor"]
    x = np.arange(4)

    fig = plt.figure(figsize=(7.25, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.35], width_ratios=[1.08, 0.92],
                          hspace=0.58, wspace=0.62)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

    # a: Hero panel, 3-year CIF at each landmark.
    main = cif[cif["window"].eq("3 years")].set_index("landmark").loc[order]
    specs = [
        ("Urethral cancer death", CANCER, "o", "cif_urethral_cancer_death_percent",
         "cif_urethral_cancer_death_ci95_lower_percent", "cif_urethral_cancer_death_ci95_upper_percent"),
        ("Other-cause death", OTHER, "s", "cif_other_cause_death_percent",
         "cif_other_cause_death_ci95_lower_percent", "cif_other_cause_death_ci95_upper_percent"),
    ]
    for label, color, marker, value_col, low_col, high_col in specs:
        y = main[value_col].to_numpy(float)
        lo = main[low_col].to_numpy(float)
        hi = main[high_col].to_numpy(float)
        ax_a.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, ms=5.4,
                      lw=2.1, capsize=2.4, capthick=0.9, elinewidth=0.9, label=label, zorder=3)
    ax_a.axvspan(2.72, 3.28, color="#ECEFF4", zorder=0)
    ax_a.text(3, 26.5, "Other-cause mortality\nbecomes predominant", ha="center", va="top",
              fontsize=7.0, color=INK, fontweight="bold")
    ax_a.annotate("", xy=(3, 15.4), xytext=(3, 10.8),
                  arrowprops=dict(arrowstyle="<->", color=MID, lw=0.8))
    ax_a.set_ylim(0, 40)
    ax_a.set_yticks([0, 10, 20, 30, 40])
    ax_a.set_ylabel("Three-year cumulative incidence (%)")
    ax_a.set_xticks(x, [f"Diagnosis\nn={main.loc['Diagnosis','n']:,}",
                        f"1-year survivor\nn={main.loc['1-year survivor','n']:,}",
                        f"3-year survivor\nn={main.loc['3-year survivor','n']:,}",
                        f"5-year survivor\nn={main.loc['5-year survivor','n']:,}"])
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.60, 0.98), ncol=2,
                handlelength=2.5, columnspacing=1.8)
    ax_a.set_title("Competing mortality shifts with time survived", loc="left", pad=8, fontweight="bold")
    style_axis(ax_a)
    panel_label(ax_a, "a", x=-0.075, y=1.04)

    # b: Robustness across risk windows.
    windows = ["1 year", "3 years", "5 years"]
    matrix_b = np.empty((3, 4))
    for i, window in enumerate(windows):
        z = cif[cif["window"].eq(window)].set_index("landmark").loc[order]
        matrix_b[i, :] = z["cif_other_cause_death_percent"] - z["cif_urethral_cancer_death_percent"]
    im_b = annotated_heatmap(ax_b, matrix_b, ["Next 1 year", "Next 3 years", "Next 5 years"],
                             ["Dx", "1 y", "3 y", "5 y"], vmax=18,
                             title="Risk-window sensitivity")
    ax_b.set_xlabel("Landmark")
    panel_label(ax_b, "b", x=-0.24, y=1.04)

    # c: Age heterogeneity in the primary 3-year window.
    age_order = ["18-59", "60-69", "70-79", "80+"]
    age = strat[strat["stratifier"].eq("Age at diagnosis")]
    matrix_c = np.empty((4, 4))
    n_c = np.empty((4, 4))
    for i, group in enumerate(age_order):
        z = age[age["group"].eq(group)].set_index("landmark").loc[order]
        matrix_c[i, :] = z["cif_other_cause_death_percent"] - z["cif_urethral_cancer_death_percent"]
        n_c[i, :] = z["n_at_landmark"]
    im_c = annotated_heatmap(ax_c, matrix_c, age_order, ["Dx", "1 y", "3 y", "5 y"], vmax=35,
                             title="Age-specific mortality balance")
    ax_c.set_xlabel("Landmark")
    ax_c.set_ylabel("Age (years)", labelpad=4)
    panel_label(ax_c, "c", x=-0.23, y=1.04)

    # Compact semantic legend avoids a colorbar competing with the forest panel.
    heat_handles = [
        Line2D([0], [0], color=CANCER, lw=5),
        Line2D([0], [0], color=OTHER, lw=5),
    ]
    fig.legend(heat_handles, ["Urethral cancer death predominates", "Other-cause death predominates"],
               loc="center", bbox_to_anchor=(0.52, 0.372), ncol=2,
               handlelength=1.7, columnspacing=1.8, fontsize=6.5)

    # d: Parsimonious adjusted cause-specific hazards among 5-year survivors.
    five = cox[cox["landmark"].eq("5-year survivor")].copy()
    cov_order = [
        "Age at diagnosis, per 10 years",
        "Male (vs female)",
        "Regional stage (vs localized)",
        "Distant stage (vs localized)",
    ]
    display_labels = ["Age, per 10 years", "Male", "Regional stage", "Distant stage"]
    ybase = np.arange(len(cov_order))[::-1]
    offsets = {"Urethral cancer death": 0.12, "Other-cause death": -0.12}
    styles = {"Urethral cancer death": (CANCER, "o"), "Other-cause death": (OTHER, "s")}
    for outcome in ["Urethral cancer death", "Other-cause death"]:
        z = five[five["outcome"].eq(outcome)].set_index("covariate").loc[cov_order]
        est = z["hazard_ratio"].to_numpy(float)
        lo = z["ci95_lower"].to_numpy(float)
        hi = z["ci95_upper"].to_numpy(float)
        color, marker = styles[outcome]
        y = ybase + offsets[outcome]
        ax_d.errorbar(est, y, xerr=[est - lo, hi - est], fmt=marker, color=color,
                      ms=4.8, lw=1.2, capsize=2.2, elinewidth=1.0, label=outcome, zorder=3)
    ax_d.axvline(1, ls="--", lw=0.9, color=MID, zorder=1)
    ax_d.set_xscale("log")
    ax_d.set_xlim(0.3, 3.6)
    ax_d.set_xticks([0.5, 1, 2, 3], labels=["0.5", "1", "2", "3"])
    ax_d.set_yticks(ybase, display_labels)
    ax_d.set_xlabel("Adjusted cause-specific hazard ratio (log scale)")
    ax_d.set_title("Adjusted cause-specific hazards after 5-year survival",
                   loc="left", pad=7, fontweight="bold")
    ax_d.legend(loc="upper right", bbox_to_anchor=(1.0, 1.08), ncol=2, handlelength=2.0)
    for yi in [0.5, 2.5]:
        ax_d.axhline(yi, color="#E7E7E7", lw=0.6, zorder=0)
    style_axis(ax_d)
    panel_label(ax_d, "d", x=-0.075, y=1.04)

    fig.suptitle("Conditional competing mortality after primary urethral carcinoma",
                 x=0.5, y=0.995, fontsize=11, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.075)
    export_figure(fig, "figure2_conditional_competing_mortality")


def make_figure_s1_subgroup_balance() -> None:
    strat = pd.read_csv(DATA_DIR / "landmark_stratified_cif_3years.csv")
    order = ["Diagnosis", "1-year survivor", "3-year survivor", "5-year survivor"]
    specifications = [
        ("Sex", ["Male", "Female"], ["Male", "Female"]),
        ("Stage", ["Localized", "Regional", "Distant", "Unknown"],
         ["Localized", "Regional", "Distant", "Unknown"]),
        ("Histology", ["Urothelial carcinoma", "Squamous cell carcinoma", "Adenocarcinoma",
                       "Epithelial NOS", "Other epithelial carcinoma"],
         ["Urothelial", "Squamous", "Adenocarcinoma", "Epithelial NOS", "Other epithelial"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.7), gridspec_kw={"height_ratios": [0.75, 1.05, 1.25]})
    images = []
    for idx, (stratifier, groups, labels) in enumerate(specifications):
        sub = strat[strat["stratifier"].eq(stratifier)]
        matrix = np.empty((len(groups), 4))
        ns = np.empty((len(groups), 4))
        for i, group in enumerate(groups):
            z = sub[sub["group"].eq(group)].set_index("landmark").loc[order]
            matrix[i, :] = z["cif_other_cause_death_percent"] - z["cif_urethral_cancer_death_percent"]
            ns[i, :] = z["n_at_landmark"]
        matrix[ns < 30] = np.nan
        im = annotated_heatmap(axes[idx], matrix, labels,
                               ["Diagnosis", "1-year survivor", "3-year survivor", "5-year survivor"],
                               vmax=35, title=f"{stratifier}", secondary=ns)
        images.append(im)
        panel_label(axes[idx], chr(ord("a") + idx), x=-0.17, y=1.03)
        if idx < 2:
            axes[idx].set_xticklabels([])
            axes[idx].set_xlabel("")
        else:
            axes[idx].set_xlabel("Landmark")
    cax = fig.add_axes([0.22, 0.055, 0.72, 0.018])
    cbar = fig.colorbar(images[-1], cax=cax, orientation="horizontal")
    cbar.set_label("Other-cause minus urethral cancer death CIF over the subsequent 3 years\n(percentage points)",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.5)
    handles = [Line2D([0], [0], color=CANCER, lw=6), Line2D([0], [0], color=OTHER, lw=6)]
    fig.legend(handles, ["Urethral cancer death predominates", "Other-cause death predominates"],
               loc="upper center", bbox_to_anchor=(0.54, 0.955), ncol=2, handlelength=1.8)
    fig.suptitle("Subgroup variation in conditional mortality balance", y=0.995,
                 fontsize=11, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.90, bottom=0.15, hspace=0.36)
    export_figure(fig, "figureS1_subgroup_mortality_balance")


def write_figure_notes() -> None:
    notes = """# Figure legends and QA notes

## Fig. 1 | Study cohort and outcome classification

Flow from the SEER*Stat tumor case listing to the patient-level analytic cohort. Adults with microscopically confirmed epithelial primary urethral carcinoma were included. For patients with more than one eligible urethral primary, the earliest eligible record was retained. Follow-up status was classified as urethral cancer death, other-cause death, alive, or death with unknown cause censored at death. The final cohort included 2,638 patients.

## Fig. 2 | Conditional competing mortality after primary urethral carcinoma

**a**, Aalen–Johansen cumulative incidence functions (CIFs) for urethral cancer death and other-cause death over the subsequent 3 years at diagnosis and among 1-, 3-, and 5-year survivors. Points show CIF estimates and error bars show nonparametric 95% confidence intervals from 1,000 bootstrap resamples. **b**, Difference between other-cause and urethral cancer death CIFs across subsequent 1-, 3-, and 5-year risk windows; positive values indicate predominance of other-cause mortality. **c**, Age-specific difference in the two CIFs over the subsequent 3 years. **d**, Cause-specific hazard ratios among 5-year survivors, adjusted for age at diagnosis, sex, and summary stage. Error bars show 95% confidence intervals. Competing deaths were treated as censored in the cause-specific Cox models. Source data are provided as aggregate CSV files.

## Supplementary Fig. 1 | Subgroup variation in conditional mortality balance

Difference between other-cause death and urethral cancer death CIFs over the subsequent 3 years, stratified by sex, summary stage, and histology. Positive values indicate predominance of other-cause mortality. Cells show the risk difference in percentage points and the number of patients at each landmark. Cells with fewer than 30 patients are suppressed and hatched. Estimates are descriptive and are conditional on survival to each landmark.

## QA record

- Backend: Python/matplotlib exclusively.
- Figure archetype: quantitative clinical composite with one hero panel.
- Final width: approximately 183 mm (double-column); height below 180 mm for the primary composite.
- Editable text: retained in SVG; TrueType embedded in PDF.
- Raster export: TIFF at 600 dpi; PNG preview at 300 dpi.
- Color integrity: outcome identity is encoded by both color and marker shape; no rainbow palette.
- Statistics: sample sizes, interval definition, model adjustment set, and competing-risk method are stated in the legends.
- Source data: all panels trace to the aggregate CSV outputs generated by the analysis script.
- Interpretation boundary: landmark contrasts are conditional associations and must not be interpreted as treatment effects or as elimination of initial-stage prognostic importance.
"""
    (FIG_DIR / "figure_legends_and_QA.md").write_text(notes, encoding="utf-8")


def main() -> None:
    required = [
        DATA_DIR / "cohort_flow.csv",
        DATA_DIR / "outcome_status.csv",
        DATA_DIR / "landmark_cif_estimates.csv",
        DATA_DIR / "landmark_stratified_cif_3years.csv",
        DATA_DIR / "landmark_cause_specific_cox_3years.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing aggregate analysis outputs: " + ", ".join(missing))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_figure1_flow()
    make_figure2_competing_mortality()
    make_figure_s1_subgroup_balance()
    write_figure_notes()


if __name__ == "__main__":
    main()
