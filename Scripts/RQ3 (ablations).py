import io
import re
import numpy as np
import pandas as pd

# 1. Load Datasets
BATADAL_path = r"Scripts\data\\RQ3 (ablations)\\3_Ablations_BATADAL_window8.csv"
SWAT_path = r"Scripts\data\\RQ3 (ablations)\Ablations_SWAT_window8.csv"
WADI_path = r"Scripts\data\\RQ3 (ablations)\Ablations_WADI_window8_MLP.csv"
skip_categories = ["Latent Fusion"]  # Only include Loss category for RQ3
try:
    df_batadal = pd.read_csv(BATADAL_path)
    df_batadal["dataset_name"] = "batadal"
    df_batadal = df_batadal[df_batadal["window_size"] == 16]

    df_swat = pd.read_csv(SWAT_path)
    df_swat["dataset_name"] = "swat"

    df_wadi = pd.read_csv(WADI_path)
    df_wadi["dataset_name"] = "wadi"

    df = pd.concat([df_batadal, df_swat, df_wadi], ignore_index=True)

except Exception as e:
    print(
        f"Loading local files failed ({e}), "
        "initializing empty fallback dataframe..."
    )
    df = pd.DataFrame()

default_losses = {
    "encoder_alpha": 1.0,
    "decoder_alpha": 1.0,
    "encoder_gamma (smooth)": 0.5,
    "decoder_gamma": 0.5,
    "encoder_lambda (sparse)": 0.5,
    "decoder_lambda": 0.5,
    "beta": 0.005,
}

# Default architecture mapping for filtering logic
default_architecture = {
    "batadal": {
        "latent_mode": "mul",
        "beta": 0,
    },
    "swat": {
        "latent_mode": "mul",
        "beta": 0.005,
    },
    "wadi": {
        "latent_mode": "mul",
        "beta": 0.005,
    },
}

# User-defined selected configurations per dataset and category
selected_config = {
    "batadal": {
        "Latent Fusion": "Latent Fusion (Mul)",
        "Architecture": "Pred (Linear)",
    },
    "swat": {
        "Latent Fusion": "Latent Fusion (Mul)",
        "Architecture": "Pred (MLP)",
    },
    "wadi": {
        "Latent Fusion": "Latent Fusion (Mul)",
        "Architecture": "w/o L-Attn",
    },
}


# 2. Define Explicit Cases
seeds = 3
ablation_cases = [
    {
        "name": "w/o Sparse",
        "is_proposed": False,
        "category": "Loss",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": 0.0,
            "decoder_lambda": 0.0,
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "w/o Smooth",
        "is_proposed": False,
        "category": "Loss",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": 0.0,
            "decoder_gamma": 0.0,
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "w/o KL",
        "is_proposed": False,
        "category": "Loss",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": 0.0,
        },
    },
    {
        "name": "Latent Fusion (Gate)",
        "is_proposed": False,
        "category": "Latent Fusion",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "gate",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "Latent Fusion (Add)",
        "is_proposed": False,
        "category": "Latent Fusion",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "add",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "Latent Fusion (Mul)",
        "is_proposed": False,
        "category": "Latent Fusion",
        "filters": {
            "latent_mode": "mul",
            "context": "linear_attn",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "w/o Orth",
        "is_proposed": False,
        "category": "Architecture",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 1,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "w/o Mixer",
        "is_proposed": False,
        "category": "Architecture",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 0,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "w/o L-Attn",
        "is_proposed": False,
        "category": "Architecture",
        "filters": {
            "context": "gate",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "Pred (MLP)",
        "is_proposed": False,
        "category": "Architecture",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "mlp",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
    {
        "name": "Pred (Linear)",
        "is_proposed": False,
        "category": "Architecture",
        "filters": {
            "context": "linear_attn",
            "latent_mode": "mul",
            "pool": "max",
            "coeff_mode": "symmetric",
            "predictor": "linear",
            "disable_orth_proj": 0,
            "temporal_mixer": 1,
            "encoder_alpha": default_losses["encoder_alpha"],
            "decoder_alpha": default_losses["decoder_alpha"],
            "encoder_gamma (smooth)": default_losses["encoder_gamma (smooth)"],
            "decoder_gamma": default_losses["decoder_gamma"],
            "encoder_lambda (sparse)": default_losses[
                "encoder_lambda (sparse)"
            ],
            "decoder_lambda": default_losses["decoder_lambda"],
            "beta": default_losses["beta"],
        },
    },
]


def set_detaulfs_per_dataset(filters, dataset_name):
    """Set default values for missing hyperparameters based on dataset."""
    for key, value in filters.items():
        if key in default_architecture[dataset_name]:
            filters[key] = default_architecture[dataset_name][key]
            #print(
            #    f"Setting default for {key} to {filters[key]} for dataset {dataset_name}"
            #)
    return filters


def filter_case(data_frame, filters, category, dataset_name):
    """Apply case hyperparameter constraints to dataframe."""
    query = data_frame

    if category == "Architecture":
        filters = set_detaulfs_per_dataset(filters, dataset_name)
    for k, v in filters.items():
        if k in query.columns:
            query = query[query[k] == v]

    return query


def is_boxed_case(dataset_name, category, case_name):
    """Determine whether a case should be boxed based on selected_config."""
    ds_key = dataset_name.lower()
    for config_ds, categories in selected_config.items():
        if config_ds.lower() == ds_key:
            target_case = categories.get(category)
            if target_case and target_case == case_name:
                return True
    return False


def get_accuracy_color_macro(val, min_val, max_val):
    """Generates 35% vibrant background color scaling for metrics."""
    if pd.isna(val) or min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return ""

    norm = (val - min_val) / (max_val - min_val)
    max_opacity = 35

    if norm > 0.5:
        pct_green = int((norm - 0.5) * 2 * 100)
        return f"\\cellcolor{{green!{pct_green}!yellow!{max_opacity}}}"
    else:
        pct_yellow = int(norm * 2 * 100)
        pct_red = 100 - pct_yellow
        return f"\\cellcolor{{yellow!{pct_yellow}!red!{max_opacity}}}"


# =========================================================
# Parameter formatting
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
# Get parameter count for a method + dataset
# =========================================================
def get_params(sub_df):

    values = sub_df["total_params"]

    if values.empty:
        return "-"

    mean_params = values.mean()

    return format_params(mean_params)



# 3. Construct Table Rows
target_metrics = [
    ("AC@1", "AC@1"),
    ("AC@5", "AC@5"),
    ("AC@10", "AC@10"),
    ("auc@10", "auc@10")
]

latex_lines = []
previous_category = None
for case in ablation_cases:

    case_name = case["name"]
    is_proposed = case["is_proposed"]
    category = case["category"]
    metric_cells = []
    if category in skip_categories:
        continue  # Skip non-target categories
    # Add section separator when category changes
    if previous_category is not None and category != previous_category:
        latex_lines.append("\\midrule")
        latex_lines.append(
            f"\\multicolumn{{10}}{{l}}{{\\textit{{{category} Ablations}}}} \\\\"
        )
        latex_lines.append("\\midrule")

    # Add category label before first group
    if previous_category is None:
        latex_lines.append(
            f"\\multicolumn{{10}}{{l}}{{\\textit{{{category} Ablations}}}} \\\\"
        )
        latex_lines.append("\\midrule")

    previous_category = category

    for ds in ["batadal", "swat", "wadi"]:

        ds_data = df[df["dataset_name"] == ds] if not df.empty else df
        sub_df = filter_case(
            ds_data, case["filters"], category, dataset_name=ds
        )
        unique_seeds = sub_df["seed"].unique()
        if len(unique_seeds) < seeds:
            print(f"Warning: {ds} - {case_name} has only {unique_seeds} unique seeds (expected {seeds}).")
        print("check complete")
        #else:
        #    print(f"Done: {ds} - {case_name}: {len(sub_df)} rows, {unique_seeds} unique seeds.")
        should_box = is_boxed_case(ds, category, case_name)
        # --- NEW CODE: Extract and add parameters for this dataset-case combination ---
        if not sub_df.empty:
            params = get_params(sub_df) # Pass sub_df to get the specific case parameters
            # Wrap in text style or leave raw depending on how get_params outputs LaTeX
            param_cell = f"\\textit{{{params}}}" 
        else:
            param_cell = "-"
        metric_cells.append(param_cell)
        # ------------------------------------------------------------------------------

        for _, m_col in target_metrics:

            case_means = []

            for c in ablation_cases:

                sub = filter_case(ds_data, c["filters"], c["category"], ds)

                if not sub.empty:

                    val = sub[m_col].mean()

                    if not pd.isna(val):
                        case_means.append(val)

            min_v = min(case_means) if case_means else np.nan

            max_v = max(case_means) if case_means else np.nan

            if not sub_df.empty and not pd.isna(sub_df[m_col].mean()):

                m_val = sub_df[m_col].mean()
                std_val = sub_df[m_col].std()

                color = get_accuracy_color_macro(
                    m_val,
                    min_v,
                    max_v,
                )

                if is_proposed:
                    val_str = f"\\textbf{{{m_val:.3f}}}"
                else:
                    val_str = f"{m_val:.3f}"

                if not pd.isna(std_val) and std_val > 0:

                    text_val = f"{val_str}{{\\tiny$\\pm${std_val:.2f}}}"

                else:

                    text_val = f"{val_str}{{\\tiny$\\pm${std_val:.2f}}}"

                if should_box:
                    cell_text = f"{color}\\fbox{{{text_val}}}"
                else:
                    cell_text = f"{color}{text_val}"

                metric_cells.append(cell_text)

            else:
                metric_cells.append("-")

    metrics_str = " & ".join(metric_cells)

    # Bold proposed framework name
    if is_proposed:
        formatted_name = f"\\textbf{{{case_name}}}"
    else:
        formatted_name = case_name

    latex_lines.append(f"{formatted_name} & {metrics_str} \\\\")


# 4. Wrap into Final Table
# 4. Wrap into Final Table
latex_table = []

latex_table.append("\\begin{table*}[t]")
latex_table.append(
    "\\caption{Case-Based Ablation Study "
    "of Architectural Components (Mean $\\pm$ SD). "
    "Boxed cells (\\fbox{...}) indicate selected configurations.}"
)
latex_table.append("\\label{tab:ablation_cases}")
latex_table.append("\\centering")

# --- Tightening Dimensions ---
latex_table.append("\\scriptsize")  # Reduce base font size from \scriptsize to \tiny
latex_table.append(
    "\\setlength{\\tabcolsep}{1.5pt}"
)  # Reduce inter-column padding (default is 6pt)
latex_table.append(
    "\\setlength{\\fboxsep}{1.0pt}"
)  # Tighten internal box padding
latex_table.append(
    "\\renewcommand{\\arraystretch}{0.95}"
)  # Slightly tighten vertical row spacing

latex_table.append(
    "\\begin{tabular}{l l *{4}{c} | l *{4}{c} | l *{4}{c}}"
)
latex_table.append("\\toprule")

latex_table.append(
    " & "
    "\\multicolumn{5}{c}{\\textbf{BATADAL}} "
    "& "
    "\\multicolumn{5}{c}{\\textbf{SWAT}} "
    "& "
    "\\multicolumn{5}{c}{\\textbf{WADI}} "
    "\\\\"
)

latex_table.append(
    "\\cmidrule(lr){2-6}" "\\cmidrule(lr){7-11}" "\\cmidrule(lr){12-16}"
)

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

# 5. Print
#print("\n".join(latex_table))


# 6. Save File
output_path = "sections/exps/RQ3 (ablations).tex"

try:

    with open(output_path, "w") as f:
        f.write("\n".join(latex_table))

    print(f"\nSuccessfully saved to {output_path}")

except Exception as e:

    print(f"\nCould not write file automatically: {e}")