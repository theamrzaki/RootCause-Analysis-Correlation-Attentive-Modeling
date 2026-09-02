import numpy as np
import pandas as pd


# =========================================================
# 1. Load data
# =========================================================
BATADAL_path = (
    r"Scripts\data\RQ1 (accuracy)\RQ_1_BATADAL_NoDownsampling_batch512_window8.csv"
)
SWAT_path = (
    r"Scripts\data\RQ1 (accuracy)\RQ_1_SWAT_NoDownsampling_batch512_window8.csv"
)

WADI_path = (
    r"Scripts\data\RQ1 (accuracy)\RQ_1_WADI_RealNoDownsampling.csv"
)

df_batadal = pd.read_csv(BATADAL_path)
df_batadal = df_batadal[df_batadal["window_size"] == 16]# only choose window size 16 for BATADAL, as it is the best performing window size for this dataset
df_swat = pd.read_csv(SWAT_path)
df_wadi = pd.read_csv(WADI_path)


df = pd.concat(
    [df_batadal,df_swat, df_wadi],
    ignore_index=True
)

def check_complete_exps(df, seeds):
    """
    check if each architecture for each dataset has the expected number of seeds (default=3)
    """
    grouped = df.groupby(["dataset_name", "architecture"])
    for (dataset, architecture), group in grouped:
        unique_seeds = group["seed"].nunique()
        if unique_seeds < seeds:
            print(f"Warning: {dataset} - {architecture} has only {unique_seeds} unique seeds (expected {seeds}).")
    print("Check complete.")
    
check_complete_exps(df, seeds=3)


# Rename proposed architecture
df["architecture"] = df["architecture"].replace({
    "vlinear": "SFlexRCA",
    "deep_mlp": "AERCA",
    "TimeMixerpp": "TimeMixer++",
    "CUTS_PLUS": "CUTS+"
})

# Remove duplicate seed entries if present
df = df.drop_duplicates(
    subset=["dataset_name", "architecture", "seed"]
)


# =========================================================
# 2. Define groups, datasets, and metrics
# =========================================================
groups = {
    "Statistical Methods": [
        "rcd",
        "baro",
        "torai"
    ],
    "Non-Causal Methods": [
        ##"Dlinear",
        ##"Fits",
        "TimeMixer++",
        "iTransformer"
    ],
    "Causal Methods": [
        "cMLP",
        "cLSTM",
        "GVAR",
        "AERCA",
        "CUTS+"
    ],

    "Proposed Method": [
        "SFlexRCA"
    ],
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

all_methods = (
    groups["Statistical Methods"]
    + groups["Causal Methods"]
    + groups["Proposed Method"]
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

    if (
        pd.isna(val)
        or min_val == max_val
        or np.isnan(min_val)
        or np.isnan(max_val)
    ):
        return ""

    norm = (
        (val - min_val)
        / (max_val - min_val)
    )

    max_opacity = 35

    if norm > 0.5:

        pct_green = int(
            (norm - 0.5) * 2 * 100
        )

        return (
            f"\\cellcolor{{"
            f"green!{pct_green}"
            f"!yellow!{max_opacity}"
            f"}}"
        )

    else:

        pct_yellow = int(
            norm * 2 * 100
        )

        pct_red = 100 - pct_yellow

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

    for method in all_methods:

        data = df[
            (df["dataset_name"] == dataset)
            & (df["architecture"] == method)
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
        & (df["architecture"] == method)
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

for group_name, method_list in groups.items():

    # -----------------------------------------------------
    # Group header
    # -----------------------------------------------------
    latex_lines.append(
        f"\\multicolumn{{12}}{{l}}{{"
        f"\\textbf{{{group_name}}}"
        f"}} \\\\"
    )

    for method in method_list:

        clean_method = (
            method.replace("_", "\\_")
        )

        #-------------------------------------------------
        # BATADAL
        #-------------------------------------------------
        batadal_params = get_params(
            "batadal",
            method
        )
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
        swat_params = get_params(
            "swat",
            method
        )

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
        wadi_params = get_params(
            "wadi",
            method
        )

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
            f"{batadal_params} & "
            f"{' & '.join(batadal_values)} & "
            f"{swat_params} & "
            f"{' & '.join(swat_values)} & "
            f"{wadi_params} & "
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
    "\\caption{Root Cause Analysis Accuracy and Model "
    "Parameters Across BATADAL and SWAT and WADI "
    "(Mean $\\pm$ SD).}"
)

latex_table.append("\\label{tab:RQ1}")

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
    "\\begin{tabular}{l l *{4}{c} | l *{4}{c} | l *{4}{c}}"
)

latex_table.append("\\toprule")

# ---------------------------------------------------------
# Dataset headers
# ---------------------------------------------------------
latex_table.append(
    " & "
    "\\multicolumn{5}{c}{\\textbf{BATADAL} \\scriptsize{(K=16, P=43, training samples=8k)} } "
    "& "
    "\\multicolumn{5}{c}{\\textbf{SWAT} \\scriptsize{(K=8, P=51, training samples=400k)}} "
    "& "
    "\\multicolumn{5}{c}{\\textbf{WADI} \\scriptsize{(K=8, P=127, training samples=1.2M)}} "
    "\\\\"
)

latex_table.append(
    "\\cmidrule(lr){2-6}" "\\cmidrule(lr){7-11}" "\\cmidrule(lr){12-16}"
)

# ---------------------------------------------------------
# Column headers
# ---------------------------------------------------------
latex_table.append(
    "\\textbf{Method} "
    "& \\textbf{Params} "
    "& \\textbf{AC@1} "
    "& \\textbf{AC@5} "
    "& \\textbf{AC@10} "
    "& \\textbf{auc@10} "
    "& \\textbf{Params} "
    "& \\textbf{AC@1} "
    "& \\textbf{AC@5} "
    "& \\textbf{AC@10} "
    "& \\textbf{auc@10} "
    "& \\textbf{Params} "
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
    "sections/exps/RQ1.tex"
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