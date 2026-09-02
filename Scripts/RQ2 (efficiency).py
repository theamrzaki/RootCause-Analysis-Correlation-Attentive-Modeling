import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# =========================================================
# Paths
# =========================================================
PI5_PATH = [
    r"Scripts\\data\\RQ2 (efficiency)\\larger_power\\RQ_2_batadal_larger.csv",
    r"Scripts\\data\\RQ2 (efficiency)\\larger_power\\RQ_2_SWaT_larger.csv",
    r"Scripts\\data\\RQ2 (efficiency)\\larger_power\\RQ_2_WADI_larger.csv"
]

PI3_PATH = [
    r"Scripts\\data\\RQ2 (efficiency)\\smaller\\RQ_2_BATADAL_RaspberryPi_Smaller.csv",
    r"Scripts\\data\\RQ2 (efficiency)\\smaller\\RQ_2_SWAT_RaspberryPi_Smaller.csv",
    r"Scripts\\data\\RQ2 (efficiency)\\smaller\\RQ_2_WADI_RaspberryPi_Smaller.csv"
]

TRAIN_PATH = [
    r"Scripts\\data\\RQ1 (accuracy)\\RQ_1_BATADAL_NoDownsampling_batch512_window8.csv",
    r"Scripts\data\\RQ1 (accuracy)\\RQ_1_SWAT_NoDownsampling_batch512_window8.csv",
    r"Scripts\data\\RQ1 (accuracy)\\RQ_1_WADI_RealNoDownsampling.csv"
]
    


OUTPUT_DIR = r"Images/RQ2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
combined_text_file_path = os.path.join(OUTPUT_DIR, "combined_plot_text.txt")
#clear the content of the combined_plot_text.txt file if it exists
if os.path.exists(combined_text_file_path):
    with open(combined_text_file_path, "w") as f:
        f.write("")

# =========================================================
# Load data
# =========================================================
df_pi5 = pd.concat([pd.read_csv(path) for path in PI5_PATH], ignore_index=True)
df_pi3 = pd.concat([pd.read_csv(path) for path in PI3_PATH], ignore_index=True)
df_train = pd.concat([pd.read_csv(path) for path in TRAIN_PATH], ignore_index=True)


def check_complete_exps(df,name, seeds):
    """
    check if each architecture for each dataset has the expected number of seeds (default=3)
    """
    grouped = df.groupby(["dataset_name", "architecture"])
    for (dataset, architecture), group in grouped:
        unique_seeds = group["seed"].unique()
        if len(unique_seeds) < seeds:
            print(f"Warning: {name} - {dataset} - {architecture} has only {unique_seeds} unique seeds (expected {seeds}).")
    
    
check_complete_exps(df_pi5, name="pi5", seeds=3)
check_complete_exps(df_pi3, name="pi3", seeds=3)
check_complete_exps(df_train, name="train", seeds=3)

print("Check complete.")

datasets = df_pi5["dataset_name"].unique().tolist()

# =========================================================
# Rename proposed architecture
# =========================================================
for data in [df_pi5, df_pi3, df_train]:
    data["architecture"] = data["architecture"].replace({
        "vlinear": "SFlexRCA",
        "deep_mlp": "AERCA",
        "TimeMixerpp": "TimeMixer++",
        "CUTS_PLUS": "CUTS+"
    })


# =========================================================
# Methods
# =========================================================
methods = [
    "GVAR",
    "AERCA",
    "cMLP",
    "cLSTM",
    "CUTS+",
    "TimeMixer++",
    "iTransformer",
    "SFlexRCA"
]


# =========================================================
# Keep only relevant methods
# =========================================================
df_pi5 = df_pi5[
    df_pi5["architecture"].isin(methods)
].copy()

df_pi3 = df_pi3[
    df_pi3["architecture"].isin(methods)
].copy()

df_train = df_train[
    df_train["architecture"].isin(methods)
].copy()


# =========================================================
# Utility: aggregate inference metrics
# =========================================================
def aggregate_runtime(df):
    if df.columns.str.contains("joules_per_sample").any():#for pi5
        stats = (
            df.groupby("architecture")
            [["avg_time", "throughput", "avg_power_w", "total_energy_j", "joules_per_sample"]]
            .agg(["mean", "std"])
            .reset_index()
        )
    else:
        stats = (
            df.groupby("architecture")
            [["avg_time", "throughput"]]
            .agg(["mean", "std"])
            .reset_index()
        )

    stats.columns = [
        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col
        for col in stats.columns
    ]

    return stats


# =========================================================
# Utility: aggregate training metrics
# =========================================================
def aggregate_training(df):

    stats = (
        df.groupby("architecture")
        [["train_avg_epoch_time", "peak_mem_mb"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    stats.columns = [
        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col
        for col in stats.columns
    ]

    return stats





# =========================================================
# Generic bar plot
# =========================================================
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =========================================================
# Updated plot_bar with conditional zoom inset
# =========================================================


def plot_bar(
    data,
    value_column,
    ylabel,
    title,
    filename,
    scale=1,
    show_zoom=False,
    zoom_last_n=3
):
    x = np.arange(len(data))
    values = (data[value_column] * scale).to_numpy()

    std_column = value_column.replace("_mean", "_std")
    if std_column in data.columns:
        errors = (data[std_column] * scale).to_numpy()
        # Replace NaN errors (e.g. single sample) with 0 for rendering
        errors = np.nan_to_num(errors, nan=0.0)
    else:
        errors = None

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        ecolor="black"
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(data["architecture"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    # Inset Zoom logic
    if show_zoom and zoom_last_n < len(data):
        ax_inset = inset_axes(
            ax, 
            width="38%", 
            height="35%", 
            loc="upper right", 
            borderpad=1.5
        )

        idx_start = len(data) - zoom_last_n
        x_inset = x[idx_start:]
        values_inset = values[idx_start:]
        errors_inset = errors[idx_start:] if errors is not None else None

        ax_inset.bar(
            x_inset,
            values_inset,
            yerr=errors_inset,
            capsize=3,
            ecolor="black"
        )

        # Inset Labels
        labels_inset = data["architecture"].iloc[idx_start:].tolist() if hasattr(data["architecture"], "iloc") else data["architecture"][idx_start:]
        ax_inset.set_xticks(x_inset)
        ax_inset.set_xticklabels(labels_inset, rotation=20, fontsize=7, ha="right")
        ax_inset.tick_params(axis='y', labelsize=7)
        ax_inset.grid(axis="y", alpha=0.25)

        # Robust Y-limit calculation using nanmax
        max_val = np.nanmax(values_inset) if len(values_inset) > 0 else 1.0
        if errors_inset is not None and len(errors_inset) > 0:
            max_val += np.nanmax(errors_inset)
        
        # Fallback if max_val is still NaN or zero
        if np.isnan(max_val) or max_val == 0:
            max_val = 1.0

        ax_inset.set_ylim(0, max_val * 1.25)

        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    # Replaced plt.tight_layout() to eliminate the layout UserWarning
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{filename}.pdf"),
        bbox_inches="tight"
    )

    if "throughput" not in filename:
        combined_text = f"Plot: {filename}\nTitle: {title}\nY-axis label: {ylabel}\n"
        combined_text += f"Architectures: {data['architecture'].tolist()}\nValues: {values.tolist()}\n"
        if errors is not None:
            combined_text += f"Std Devs: {errors.tolist()}\n"
        combined_text += "\n" + "-" * 40 + "\n"
        with open(combined_text_file_path, "a") as f:
            f.write(combined_text)

    plt.close(fig)

for dataset in tqdm(datasets, desc="Datasets", unit="dataset"):
    df_pi5_dataset = df_pi5[df_pi5["dataset_name"] == dataset]
    df_pi3_dataset = df_pi3[df_pi3["dataset_name"] == dataset]
    df_train_dataset = df_train[df_train["dataset_name"] == dataset]

    # =========================================================
    # Aggregate
    # =========================================================
    pi5_stats = aggregate_runtime(df_pi5_dataset)
    pi3_stats = aggregate_runtime(df_pi3_dataset)
    train_stats = aggregate_training(df_train_dataset)


    # =========================================================
    # Consistent method ordering
    # =========================================================
    for data in [pi5_stats, pi3_stats, train_stats]:

        data["architecture"] = pd.Categorical(
            data["architecture"],
            categories=methods,
            ordered=True
        )

        data.sort_values(
            "architecture",
            inplace=True
        )

    # =========================================================
    # 1. Inference Time — Raspberry Pi 3
    # =========================================================
    plot_bar(
        pi3_stats,
        value_column="avg_time_mean",
        ylabel="Inference Time (ms/sample)",
        title=f"Inference Time — Raspberry Pi 3 — {dataset}",
        filename=f"{dataset}_pi3_inference_time",
        scale=1000,
        show_zoom=True,  # Enable zoom inset for throughput,
        zoom_last_n=3  # Show last 3 architectures in the zoomed inset
    )


    # =========================================================
    # 2. Inference Time — Raspberry Pi 5
    # =========================================================
    plot_bar(
        pi5_stats,
        value_column="avg_time_mean",
        ylabel="Inference Time (ms/sample)",
        title=f"Inference Time — Raspberry Pi 5 — {dataset}",
        filename=f"{dataset}_pi5_inference_time",
        scale=1000,
        show_zoom=True,  # Enable zoom inset for throughput,
        zoom_last_n=3  # Show last 3 architectures in the zoomed inset
    )


    # =========================================================
    # 3. Throughput — Raspberry Pi 3
    # =========================================================
    plot_bar(
        pi3_stats,
        value_column="throughput_mean",
        ylabel="Throughput (samples/s)",
        title=f"Inference Throughput — Raspberry Pi 3 — {dataset}",
        filename=f"{dataset}_pi3_throughput",
        scale=1
    )


    # =========================================================
    # 4. Throughput — Raspberry Pi 5
    # =========================================================
    plot_bar(
        pi5_stats,
        value_column="throughput_mean",
        ylabel="Throughput (samples/s)",
        title=f"Inference Throughput — Raspberry Pi 5 — {dataset}",
        filename=f"{dataset}_pi5_throughput",
        scale=1,

    )


    # =========================================================
    # 5. Average Training Time per Epoch
    # =========================================================
    plot_bar(
        train_stats,
        value_column="train_avg_epoch_time_mean",
        ylabel="Training Time (s/epoch)",
        title=f"Average Training Time per Epoch — {dataset}",
        filename=f"{dataset}_training_time",
        scale=1
    )


    # =========================================================
    # 6. Peak Training Memory
    # =========================================================
    plot_bar(
        train_stats,
        value_column="peak_mem_mb_mean",
        ylabel="Peak Memory (MB)",
        title=f"Peak Training Memory — {dataset}",
        filename=f"{dataset}_training_memory",
        scale=1
    )



    #avg_power_w,total_energy_j,joules_per_sample
    #========================================================
    # 7. Total Energy Consumption
    # =========================================================
    plot_bar(
        pi5_stats,
        value_column="total_energy_j_mean",
        ylabel="Total Energy Consumption (J)",
        title=f"Total Energy Consumption — Raspberry Pi 5 — {dataset}",
        filename=f"{dataset}_pi5_total_energy",
        scale=1,
        show_zoom=True,  # Enable zoom inset for throughput,
        zoom_last_n=3  # Show last 3 architectures in the zoomed inset
    )


    #========================================================
    # 8. Energy per Sample
    # =========================================================
    plot_bar(
        pi5_stats,
        value_column="joules_per_sample_mean",
        ylabel="Energy per Sample (J/sample)",
        title=f"Energy per Sample — Raspberry Pi 5 — {dataset}",
        filename=f"{dataset}_pi5_energy_per_sample",
        scale=1,
        show_zoom=True,  # Enable zoom inset for throughput,
        zoom_last_n=3  # Show last 3 architectures in the zoomed inset
    )