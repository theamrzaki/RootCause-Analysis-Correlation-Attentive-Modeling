import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "wadi"  # Change to your dataset name if needed
window_size = 8  # Change to your window size if needed
num_vars = 127  # Change to your number of variables if needed

# Replace with your actual 128x128 NumPy array or matrix
matrix = np.load(
    f"/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/datasets/{dataset}/window_{window_size}_vars_{num_vars}/orth_transform_meta/swat_q_matrix_lag{window_size}.npy"
)
plt.figure(figsize=(10, 8))
sns.heatmap(
    matrix,
    cmap='viridis',        # Choices: 'viridis', 'magma', 'coolwarm', 'plasma'
    cbar=True,             # Displays colorbar
    xticklabels=16,        # Show axis ticks every 16 units to keep axes clean
    yticklabels=16
)
plt.title("128x128 Matrix Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()


plt.savefig(
    f"/home/db2003/Desktop/Amr/(TSE) RootCause-Analysis-Correlation-Attentive-Modeling/{dataset}_window_{window_size}_vars_{num_vars}_heatmap.png"
)