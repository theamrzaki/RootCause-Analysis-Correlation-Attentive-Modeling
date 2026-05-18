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

# Unified Typography Configuration
x_ticks_fontsize = 20
y_ticks_fontsize = 20
legend_fontsize = 16
window_size_label_fontsize = 20
params_label_fontsize = 20

# Optional configurations for clarity
title_fontsize = params_label_fontsize + 2
annotation_text_fontsize = legend_fontsize - 2

# =========================================================
# Ground-truth root causes
# =========================================================
labels = np.load(os.path.join(CASE_DIR, "labels.npy"))
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

plt.figure(figsize=(12, 5))

# normal variables
plt.plot(
    x,
    scores,
    marker='o',
    linewidth=1.5,
    alpha=0.7,
)

# highlight GT root causes
plt.scatter(
    TRUE_ROOT_CAUSES,
    scores[TRUE_ROOT_CAUSES],
    s=150,  # Scaled up slightly to balance large labels
    marker='o',
    color='red',
    label='Ground Truth Root Cause',
    zorder=5
)

# Determine an offset based on data scale
y_offset = max(scores) * 0.03  

for idx in TRUE_ROOT_CAUSES:
    plt.text(
        idx,
        scores[idx] + y_offset,
        f"{idx}",
        fontsize=annotation_text_fontsize,
        fontweight='bold',
        color='darkred',
        ha='center',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        zorder=6
    )

#plt.title("Raw Residual Anomaly Signal", fontsize=title_fontsize, fontweight='bold', pad=15)
plt.xlabel("Variable Index", fontsize=params_label_fontsize, labelpad=10)
plt.ylabel("Residual Magnitude", fontsize=params_label_fontsize, labelpad=10)

plt.xticks(fontsize=x_ticks_fontsize)
plt.yticks(fontsize=y_ticks_fontsize)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=legend_fontsize, loc='upper right')
plt.tight_layout()

plt.savefig(os.path.join(CASE_DIR, "plot_residual.pdf"), dpi=300)
plt.show()

# =========================================================
# 2. Top-k ranking visualization
# =========================================================
TOP_K = 10

ranked_scores = z_scores[0][ranking[:TOP_K]]
ranked_vars = ranking[:TOP_K]

plt.figure(figsize=(12, 6))  # Expanded slightly to host larger labels comfortably

bar_colors = []
plot_labels = []

for var in ranked_vars:
    if var in TRUE_ROOT_CAUSES:
        plot_labels.append(f"{var}*")
        bar_colors.append('#d62728')  
    else:
        plot_labels.append(str(var))
        bar_colors.append('#1f77b4')  

bars = plt.bar(range(TOP_K), ranked_scores, color=bar_colors, edgecolor='none', alpha=0.85)

y_offset_bar = max(ranked_scores) * 0.015  

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + y_offset_bar,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=annotation_text_fontsize,
        fontweight='semibold',
        color='#333333'
    )

plt.xticks(range(TOP_K), plot_labels, fontsize=x_ticks_fontsize, fontweight='medium')
plt.yticks(fontsize=y_ticks_fontsize)

plt.xlabel("Ranked Variables (* = Ground Truth Root Cause)", fontsize=params_label_fontsize, labelpad=12)
plt.ylabel("Anomaly Score (Z-Score)", fontsize=params_label_fontsize, labelpad=12)
#plt.title("Top-K RCA Ranking (Root Cause Diagnostics)", fontsize=title_fontsize, fontweight='bold', pad=15)

plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
plt.gca().set_axisbelow(True)

# Expand y-limit to prevent text clipping given the larger font sizes
plt.ylim(0, max(ranked_scores) * 1.20)

plt.tight_layout()
plt.savefig(os.path.join(CASE_DIR, "plot_topk.pdf"), dpi=300)
plt.show()

print("\nSaved figures:")
print("- plot_residual.pdf")
print("- plot_topk.pdf")
print("Top-10 ranking:", ranking[:10])
print("Top-10 z-scores:", z_scores[0][ranking[:10]])
print("True scores:", z_scores[0][TRUE_ROOT_CAUSES])

# =========================================================
# 3. Automated Text Report Generation
# =========================================================
gt_in_top_k = [int(v) for v in ranked_vars if v in TRUE_ROOT_CAUSES]
hit_rate = len(gt_in_top_k) / len(TRUE_ROOT_CAUSES) if len(TRUE_ROOT_CAUSES) > 0 else 0.0

gt_ranks_text = ""
for gt in TRUE_ROOT_CAUSES:
    positions = np.where(ranking == gt)[0]
    rank_str = f"Rank {positions[0] + 1}" if len(positions) > 0 else "Not Found"
    gt_ranks_text += f"*   **Variable {gt}:** {rank_str} (Residual: {scores[gt]:.4f})\n"

table_rows = ""
for i in range(TOP_K):
    var = ranked_vars[i]
    is_gt = "Yes (★)" if var in TRUE_ROOT_CAUSES else "No"
    table_rows += f"| {i+1} | {var} | {ranked_scores[i]:.4f} | {is_gt} |\n"

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

report_path = os.path.join(CASE_DIR, "rca_diagnostic_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Success! Report saved cleanly to: {report_path}")