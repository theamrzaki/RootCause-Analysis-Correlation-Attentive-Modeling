import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
SWAT_path = r"Scripts\data\\RQ4 (sensitivity)\Ablations_swat_windows.csv"
WADI_path = r"Scripts\data\\RQ4 (sensitivity)\Ablations_wadi_windows.csv"
BATADAL_path = (
    r"Scripts\data\\RQ4 (sensitivity)\\4_Ablations_batadal_windows_beta_0.csv"
)

df_swat = pd.read_csv(SWAT_path)
df_wadi = pd.read_csv(WADI_path)
df_batadal = pd.read_csv(BATADAL_path)
df = pd.concat([df_swat, df_wadi, df_batadal], ignore_index=True)

combined_text_file_path = os.path.join("Images/RQ4/", "combined_plot_text.txt")
#clear the content of the combined_plot_text.txt file if it exists
if os.path.exists(combined_text_file_path):
    with open(combined_text_file_path, "w") as f:
        f.write("")

def check_complete_exps(df, seeds):
    """
    check if each window_size for each dataset has the expected number of seeds (default=3)
    """
    grouped = df.groupby(["dataset_name", "window_size"])
    for (dataset, window_size), group in grouped:
        unique_seeds = group["seed"].nunique()
        if unique_seeds < seeds:
            print(f"Warning: {name} - {dataset} - {window_size} has only {unique_seeds} unique seeds (expected {seeds}).")
    
    
check_complete_exps(df, seeds=3)
print("Check complete.")

# Rename proposed architecture
df["architecture"] = df["architecture"].replace({"vlinear": "SFlexRCA"})

# ---------------------------------------------------------
# Keep only proposed configuration
# ---------------------------------------------------------
df = df[df["architecture"] == "SFlexRCA"].copy()

# ---------------------------------------------------------
# Define target window size per dataset to highlight
# ---------------------------------------------------------
highlight_windows = {
    "batadal": 16,
    "swat": 8,
    "wadi": 8,
}

# ---------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------
selected_metrics = ["AC@1", "AC@5", "AC@10"]

loss_stats = (
    df.groupby(["dataset_name", "window_size"])[selected_metrics]
    .agg(["mean", "std"])
    .reset_index()
)

# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------
os.makedirs("Images/RQ4", exist_ok=True)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
for dataset in loss_stats["dataset_name"].unique():

    data = loss_stats[loss_stats["dataset_name"] == dataset].sort_values(
        "window_size"
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    # Highlight specific window size if defined for the dataset
    ds_key = dataset.lower()
    if ds_key in highlight_windows:
        target_win = highlight_windows[ds_key]

        # Draw a subtle shaded region behind the selected window
        ax.axvspan(
            target_win - 0.5,
            target_win + 0.5,
            color="grey",
            alpha=0.15,
            zorder=0,
            label=f"Selected ($w={target_win}$)",
        )

        # Optional: Vertical dashed line marker
        ax.axvline(
            x=target_win, color="gray", linestyle="--", alpha=0.6, zorder=1
        )

    for metric in selected_metrics:
        mean_values = data[(metric, "mean")]

        ax.errorbar(
            data["window_size"],
            mean_values,
            yerr=data[(metric, "std")],
            marker="o",
            linewidth=2,
            label=metric,
            capsize=5,
            zorder=3,
        )

    combined_text = ""
    combined_text += f"Dataset: {dataset}\n"
    for metric in selected_metrics:
        combined_text += f"{metric}:\n"
        for _, row in data.iterrows():
            combined_text += f"  Window Size {row['window_size']}: Mean = {row[(metric, 'mean')]:.4f}, Std = {row[(metric, 'std')]:.4f}\n"
    combined_text += "\n"
    with open(combined_text_file_path, "a") as f:
        f.write(combined_text)

    ax.set_xticks(data["window_size"])
    ax.set_xlabel(r"Window size ($w$)")
    ax.set_ylabel("RCA Accuracy")
    ax.set_title(f"{dataset.upper()}: RCA Accuracy vs. Window Size")

    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    output_path = f"Images/RQ4/{dataset}_accuracy_vs_window_size.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved plot for {dataset} at {output_path}")
    plt.close()