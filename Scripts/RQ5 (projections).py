import io
import re
import numpy as np
import pandas as pd

# 1. Load Datasets
BATADAL_path = r"Scripts\data\\RQ5 (projections)\\4_Ablations_Projections_batadal.csv"
SWAT_path = r"Scripts\data\\RQ5 (projections)\\4_Ablations_Projections_swat.csv"
WADI_path = r"Scripts\data\\RQ5 (projections)\\4_Ablations_Projections_wadi.csv"

df_batadal = pd.read_csv(BATADAL_path)
df_swat = pd.read_csv(SWAT_path)
df_wadi = pd.read_csv(WADI_path)

df = pd.concat(
    [df_batadal,df_swat, df_wadi],
    ignore_index=True
)

def check_complete_exps(df, seeds):
    """
    check if each transformation for each dataset has the expected number of seeds (default=3)
    """
    grouped = df.groupby(["dataset_name", "transformation"])
    for (dataset, transformation), group in grouped:
        unique_seeds = group["seed"].nunique()
        if unique_seeds < seeds:
            print(f"Warning: {dataset} - {transformation} has only {unique_seeds} unique seeds (expected {seeds}).")
    print("Check complete.")
    
check_complete_exps(df, seeds=3)


# Rename proposed architecture
df["architecture"] = df["architecture"].replace({
    "vlinear": "SFlexRCA"
})

df["transformation"] = df["transformation"].replace({
    "orthogonal": "orthogonal (proposed)"
})

# =========================================================
# 2. Define groups, datasets, and metrics
# =========================================================
projections = {
    "Baselines": [
        "none",
        "learned"
    ],
    "Static Projections": [
        "legendre",
        "laguerre",
        "chebyshev",
        "hermite",
        "fourier"
    ],
    "Dataset Specific Projection": [
        "orthogonal (proposed)"
    ]
}

datasets = [
    "batadal",
    "swat",
    "wadi"
]

metrics = [
    ("AC@1", "AC@1"),
    ("AC@5", "AC@5"),
    ("AC@10", "AC@10"),
    ("auc@10", "auc@10")
]
#mrr,hr@1,hr@3,hr@5,hr@10,auc@10,

all_projections = (
    projections["Baselines"]
    + projections["Static Projections"]
    + projections["Dataset Specific Projection"]
)


# =========================================================
# 3. Parameter formatting
# =========================================================
def format_params(p):
    """
    Format parameter counts:
        10.7M
        127.6K
        0
    """

    if pd.isna(p) or p == 0:
        return "0"

    if p >= 1e6:
        return f"{p / 1e6:.1f}M"

    if p >= 1e3:
        return f"{p / 1e3:.1f}K"

    return str(int(p))


# =========================================================
# 4. Accuracy color
# =========================================================
def get_accuracy_color_macro(
    val,
    min_val,
    max_val
):
    """
    Generate a subtle red-yellow-green background
    according to the relative performance.
    """

    if pd.isna(val) or np.isnan(min_val) or np.isnan(max_val):
        return ""

    # MODIFICATION 1: If min_val == max_val, map to 0.5 (neutral yellow)
    if min_val == max_val:
        norm = 0.5
    else:
        norm = (val - min_val) / (max_val - min_val)

    max_opacity = 35

    if norm >= 0.5:
        pct_green = int((norm - 0.5) * 2 * 100)
        return (
            f"\\cellcolor{{"
            f"green!{pct_green}"
            f"!yellow!{max_opacity}"
            f"}}"
        )
    else:
        pct_yellow = int(norm * 2 * 100)
        return (
            f"\\cellcolor{{"
            f"yellow!{pct_yellow}"
            f"!red!{max_opacity}"
            f"}}"
        )


# =========================================================
# 5. Get parameter count for a method + dataset
# =========================================================
def get_params(dataset, method):

    values = df[
        (df["dataset_name"] == dataset)
        & (df["architecture"] == method)
    ]["total_params"]

    if values.empty:
        return "-"

    mean_params = values.mean()

    return format_params(mean_params)


# =========================================================
# 6. Get min/max for a dataset + metric
# =========================================================
def get_metric_range(dataset, metric):

    values = []

    for method in all_projections:

        data = df[
            (df["dataset_name"] == dataset)
            & (df["transformation"] == method)
        ][metric]

        if not data.empty:

            mean_val = data.mean()

            if not pd.isna(mean_val):
                values.append(mean_val)

    if not values:
        return np.nan, np.nan

    return (
        min(values),
        max(values)
    )


# =========================================================
# 7. Format one accuracy cell
# =========================================================
def format_accuracy_cell(
    dataset,
    method,
    metric
):

    data = df[
        (df["dataset_name"] == dataset)
        & (df["transformation"] == method)
    ][metric]

    if data.empty:
        return "-"

    mean_val = data.mean()

    if pd.isna(mean_val):
        return "-"

    std_val = data.std()

    min_val, max_val = get_metric_range(
        dataset,
        metric
    )

    color_prefix = get_accuracy_color_macro(
        mean_val,
        min_val,
        max_val
    )

    if pd.isna(std_val):
        std_val = 0.0

    return (
        f"{color_prefix}"
        f"{mean_val:.3f}"
        f"{{\\tiny$\\pm${std_val:.2f}}}"
    )


# =========================================================
# 8. Build table rows
# =========================================================
latex_lines = []

for group_name, projection_list in projections.items():

    # -----------------------------------------------------
    # Group header
    # -----------------------------------------------------
    latex_lines.append(
        f"\\multicolumn{{1}}{{l}}{{"
        f"\\textbf{{{group_name}}}"
        f"}} \\\\"
    )

    for method in projection_list:

        clean_method = (
            method.replace("_", "\\_")
        )

        #-------------------------------------------------
        # BATADAL
        #-------------------------------------------------
        batadal_values = [
            format_accuracy_cell(
                "batadal",
                method,
                metric
            )
            for _, metric in metrics
        ]

        # -------------------------------------------------
        # SWAT
        # -------------------------------------------------
        swat_values = [
            format_accuracy_cell(
                "swat",
                method,
                metric
            )
            for _, metric in metrics
        ]

        # -------------------------------------------------
        # WADI
        # -------------------------------------------------
        wadi_values = [
            format_accuracy_cell(
                "wadi",
                method,
                metric
            )
            for _, metric in metrics
        ]

        # -------------------------------------------------
        # Construct row
        # -------------------------------------------------
        row = (
            f"{clean_method} & "
            f"{' & '.join(batadal_values)} & "
            f"{' & '.join(swat_values)} & "
            f"{' & '.join(wadi_values)} \\\\"
        )

        latex_lines.append(row)

    # Separator after each method group
    latex_lines.append("\\midrule")


# Remove final \midrule
if latex_lines and latex_lines[-1] == "\\midrule":
    latex_lines.pop()


# =========================================================
# 9. Build complete LaTeX table
# =========================================================
latex_table = []

latex_table.append("\\begin{table*}[t]")

latex_table.append(
    "\\caption*{Root Cause Analysis Accuracy Across "
    "Different Projections For BATADAL and SWAT and WADI "
    "(Mean $\\pm$ SD).}"
)

latex_table.append("\\label{tab:projections_metrics}")

latex_table.append("\\centering")

# --- Tightening Dimensions ---
latex_table.append("\\scriptsize")  # Drops font size down to fit all 13 columns
latex_table.append(
    "\\setlength{\\tabcolsep}{2.0pt}"
)  # Reduces column padding (default is 6pt)
latex_table.append(
    "\\renewcommand{\\arraystretch}{0.92}"
)  # Compresses row height slightly
latex_table.append(
    "\\setlength{\\fboxsep}{1.0pt}"
)  # Tightens box padding if highlight borders are used

# ---------------------------------------------------------
# Column format:
# Method + 3 x (Params + AC@1 + AC@5 + AC@10) = 13 columns
# ---------------------------------------------------------
latex_table.append(
    "\\begin{tabular}{l *{4}{c} !{\\vrule width 1.2pt} *{4}{c} !{\\vrule width 1.2pt} *{4}{c}}"
)

latex_table.append("\\toprule")

# ---------------------------------------------------------
# Dataset headers
# ---------------------------------------------------------
latex_table.append(
    " & "
    "\\multicolumn{4}{c}{\\textbf{BATADAL}  } "
    "& "
    "\\multicolumn{4}{c}{\\textbf{SWAT} } "
    "& "
    "\\multicolumn{4}{c}{\\textbf{WADI} } "
    "\\\\"
)

latex_table.append(
    "\\cmidrule(lr){2-5}" "\\cmidrule(lr){6-9}" "\\cmidrule(lr){10-13}"
)

# ---------------------------------------------------------
# Column headers
# ---------------------------------------------------------
latex_table.append(
    "\\textbf{Method} "
    "& \\textbf{AC@1} "
    "& \\textbf{AC@5} "
    "& \\textbf{AC@10} "
    "& \\textbf{auc@10} "
    "& \\textbf{AC@1} "
    "& \\textbf{AC@5} "
    "& \\textbf{AC@10} "
    "& \\textbf{auc@10} "
    "& \\textbf{AC@1} "
    "& \\textbf{AC@5} "
    "& \\textbf{AC@10} "
    "& \\textbf{auc@10} "
    "\\\\"
)

latex_table.append("\\midrule")

latex_table.extend(latex_lines)

latex_table.append("\\bottomrule")

latex_table.append("\\end{tabular}")

latex_table.append("\\end{table*}")


# =========================================================
# 10. Print
# =========================================================
latex_output = "\n".join(
    latex_table
)

#print(latex_output)


# =========================================================
# 11. Save
# =========================================================
output_file = (
    "sections/exps/RQ5 (projections).tex"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(latex_output)

print(
    f"\nLaTeX table saved to {output_file}"
)