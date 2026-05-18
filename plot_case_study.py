# SFlexRCA Case Study Plotting Pipeline

import os
import numpy as np
import matplotlib.pyplot as plt


CASE_DIR = "/home/db2003/Desktop/Amr/RootCause-Analysis-Correlation-Attentive-Modeling/saved_models/case_study"


# =========================================================
# Load artifacts
# =========================================================
residual = np.load(os.path.join(CASE_DIR, "residual.npy"))
z_scores = np.load(os.path.join(CASE_DIR, "z_scores.npy"))
z_pot = np.load(os.path.join(CASE_DIR, "z_pot.npy"))
ranking = np.load(os.path.join(CASE_DIR, "ranking.npy"))

print("Residual shape:", residual.shape)
print("Z-score shape:", z_scores.shape)
print("POT shape:", z_pot.shape)
print("Ranking shape:", ranking.shape)

x_ticks_fontsize = 20
y_ticks_fontsize = 20
legend_fontsize = 16
window_size_label_fontsize = 20
params_label_fontsize = 20
# =========================================================
# Optional: define ground-truth root causes manually
# =========================================================
labels = np.load(
    os.path.join(CASE_DIR, "labels.npy")
)

TRUE_ROOT_CAUSES = np.where(labels[0] == 1)[0]
print("Ground Truth Root Causes:", TRUE_ROOT_CAUSES)


# =========================================================
# Helper
# =========================================================
def highlight_gt(ax):
    for idx in TRUE_ROOT_CAUSES:
        ax.axvline(idx, linestyle='--', alpha=0.5)


# =========================================================
# 1. Raw residual signal
# =========================================================
scores = residual[0]
x = np.arange(len(scores))

plt.figure(figsize=(12, 4))

# normal variables
plt.plot(
    x,
    scores,
    marker='o',
    linewidth=1.5,
    alpha=0.7
)

# highlight GT root causes
plt.scatter(
    TRUE_ROOT_CAUSES,
    scores[TRUE_ROOT_CAUSES],
    s=120,
    marker='o',
    color='red',
    label='Ground Truth Root Cause',
    zorder=5
)

# Determine an offset based on your data scale (e.g., 2% to 5% of the max score)
y_offset = max(scores) * 0.03  

for idx in TRUE_ROOT_CAUSES:
    plt.text(
        idx,
        scores[idx] + y_offset,  # Manually shift the text upward
        f"{idx}",
        fontsize=10,
        fontweight='bold',
        color='darkred',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        zorder=6
    )

plt.title("Raw Residual Anomaly Signal")
plt.xlabel("Variable Index")
plt.ylabel("Residual Magnitude")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(CASE_DIR, "plot_residual.pdf"),
    dpi=300
)

plt.show()

# =========================================================
# 2. Z-score normalized anomaly scores
# =========================================================
#plt.figure(figsize=(12, 4))
#plt.plot(z_scores[0], marker='o')
#highlight_gt(plt.gca())
#
#plt.title("Normalized RCA Scores (Z-score)")
#plt.xlabel("Variable Index")
#plt.ylabel("Normalized Score")
#plt.grid(True, alpha=0.3)
#plt.tight_layout()
#plt.savefig(os.path.join(CASE_DIR, "plot_zscores.pdf"), dpi=300)
##plt.show()


# =========================================================
# 3. POT-filtered anomaly scores
# =========================================================
#plt.figure(figsize=(12, 4))
#plt.bar(np.arange(len(z_pot[0])), z_pot[0])
#highlight_gt(plt.gca())
#
#plt.title("POT-Filtered RCA Scores")
#plt.xlabel("Variable Index")
#plt.ylabel("POT Score")
#plt.grid(True, alpha=0.3)
#plt.tight_layout()
#plt.savefig(os.path.join(CASE_DIR, "plot_pot.pdf"), dpi=300)
##plt.show()


# =========================================================
# 4. Top-k ranking visualization
# =========================================================
TOP_K = 10

ranked_scores = z_scores[0][ranking[:TOP_K]]
ranked_vars = ranking[:TOP_K]

plt.figure(figsize=(10, 5))

# 1. Generate dynamic colors and labels based on Ground Truth status
bar_colors = []
labels = []

for var in ranked_vars:
    if var in TRUE_ROOT_CAUSES:
        labels.append(f"{var}*")
        bar_colors.append('#d62728')  # Distinct crimson/red for GT root causes
    else:
        labels.append(str(var))
        bar_colors.append('#1f77b4')  # Clean standard blue for others

# 2. Plot the bars with their specific color mapping
bars = plt.bar(range(TOP_K), ranked_scores, color=bar_colors, edgecolor='none', alpha=0.85)

# 3. Add text labels on top of EACH bar for quick value checking
# Determine a safe vertical offset based on the maximum score
y_offset = max(ranked_scores) * 0.015  

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + y_offset,
        f"{height:.2f}",  # Formats score to 2 decimal places
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='semibold',
        color='#333333'
    )

# 4. Aesthetics & Labeling
plt.xticks(range(TOP_K), labels, fontsize=10, fontweight='medium')
plt.xlabel("Ranked Variables (* = Ground Truth Root Cause)", fontsize=11, labelpad=10)
plt.ylabel("Anomaly Score (Z-Score)", fontsize=11)
plt.title("Top-K RCA Ranking (Root Cause Diagnostics)", fontsize=13, fontweight='bold', pad=15)

# Clean up gridlines (only horizontal lines make sense for bar charts)
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars sit in front of grid lines
plt.gca().set_axisbelow(True)

# Expand y-limit slightly so top text annotations don't get clipped by the frame
plt.ylim(0, max(ranked_scores) * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(CASE_DIR, "plot_topk.pdf"), dpi=300)
plt.show()

# =========================================================
# 5. Combined RCA pipeline figure (paper-ready)
# =========================================================
#fig, axes = plt.subplots(3, 1, figsize=(12, 10))
#
## Residual
#axes[0].plot(residual[0], marker='o')
#for idx in TRUE_ROOT_CAUSES:
#    axes[0].axvline(idx, linestyle='--', alpha=0.5)
#axes[0].set_title("Residual Signal")
#axes[0].grid(True, alpha=0.3)
#
## Z-score
#axes[1].plot(z_scores[0], marker='o')
#for idx in TRUE_ROOT_CAUSES:
#    axes[1].axvline(idx, linestyle='--', alpha=0.5)
#axes[1].set_title("Normalized RCA Scores")
#axes[1].grid(True, alpha=0.3)
#
## POT
#axes[2].bar(np.arange(len(z_pot[0])), z_pot[0])
#for idx in TRUE_ROOT_CAUSES:
#    axes[2].axvline(idx, linestyle='--', alpha=0.5)
#axes[2].set_title("POT-Filtered RCA Scores")
#axes[2].grid(True, alpha=0.3)
#
#for ax in axes:
#    ax.set_xlabel("Variable Index")
#    ax.set_ylabel("Score")
#
#plt.tight_layout()
#plt.savefig(os.path.join(CASE_DIR, "plot_pipeline_overview.pdf"), dpi=300)
#plt.show()


print("\nSaved figures:")
print("- plot_residual.pdf")
print("- plot_zscores.pdf")
print("- plot_pot.pdf")
print("- plot_topk.pdf")
print("- plot_pipeline_overview.pdf")
print("Top-10 ranking:", ranking[:10])
print("Top-10 z-scores:", z_scores[0][ranking[:10]])
print("True scores:", z_scores[0][TRUE_ROOT_CAUSES])




# =========================================================
# 3. Automated Text Report Generation
# =========================================================
# Parse statistics dynamically from the data
gt_in_top_k = [int(v) for v in ranked_vars if v in TRUE_ROOT_CAUSES]
hit_rate = len(gt_in_top_k) / len(TRUE_ROOT_CAUSES) if len(TRUE_ROOT_CAUSES) > 0 else 0.0

# Find localization ranks for each Ground Truth variable
gt_ranks_text = ""
for gt in TRUE_ROOT_CAUSES:
    positions = np.where(ranking == gt)[0]
    rank_str = f"Rank {positions[0] + 1}" if len(positions) > 0 else "Not Found"
    gt_ranks_text += f"*   **Variable {gt}:** {rank_str} (Residual: {scores[gt]:.4f})\n"

# Build the Top-K table markdown string
table_rows = ""
for i in range(TOP_K):
    var = ranked_vars[i]
    is_gt = "Yes (★)" if var in TRUE_ROOT_CAUSES else "No"
    table_rows += f"| {i+1} | {var} | {ranked_scores[i]:.4f} | {is_gt} |\n"

# Construct the full text report string
report_content = f"""# Root Cause Analysis (RCA) Diagnostic Report

## Executive Summary
*   **Total Ground Truth Root Causes:** {len(TRUE_ROOT_CAUSES)} (Variables: {list(TRUE_ROOT_CAUSES)})
*   **Ground Truth Detected in Top-{TOP_K}:** {len(gt_in_top_k)} / {len(TRUE_ROOT_CAUSES)} ({hit_rate * 100:.1f}% Hit Rate)
*   **Highest Scoring Variable:** Index {ranking[0]} (Z-Score: {z_scores[0][ranking[0]]:.4f})

---

## Ground Truth Localization Performance
Below is the precise positioning of the actual root causes within the overall framework ranking:
{gt_ranks_text}
---

## Top-{TOP_K} Diagnostic Ranking Breakdown
This table matches the data points visualized within `plot_topk.pdf`:

| Rank | Variable Index | Anomaly Score (Z-Score) | Is Ground Truth? |
| :--- | :------------- | :---------------------- | :--------------- |
{table_rows}
---

## Signal Metrics Summary
*   **Raw Residual Max Magnitude:** {np.max(scores):.4f} (at Index {np.argmax(scores)})
*   **Raw Residual Mean:** {np.mean(scores):.4f}
*   **Raw Residual Standard Deviation:** {np.std(scores):.4f}
*   **Anomaly Score Max (Z-Score):** {np.max(z_scores[0]):.4f}
"""

# Save the report to a text file
report_path = os.path.join(CASE_DIR, "rca_diagnostic_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Success! Report saved cleanly to: {report_path}")