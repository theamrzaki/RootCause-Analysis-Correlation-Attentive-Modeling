# Root Cause Analysis (RCA) Diagnostic Report

## Executive Summary
*   **Total Ground Truth Root Causes:** 7 (Variables: [0, 8, 9, 11, 12, 13, 14])
*   **Ground Truth Detected in Top-10:** 3 / 7 (42.9% Hit Rate)
*   **Highest Scoring Variable:** Index 31 (Z-Score: 2.5905)

---

## Ground Truth Localization Performance
Below is the precise positioning of the actual root causes within the overall framework ranking:
*   **Variable 0:** Rank 2 (Residual: -0.9483)
*   **Variable 8:** Rank 3 (Residual: -1.4336)
*   **Variable 9:** Rank 34 (Residual: 4.6731)
*   **Variable 11:** Rank 35 (Residual: 0.8629)
*   **Variable 12:** Rank 8 (Residual: -0.5270)
*   **Variable 13:** Rank 36 (Residual: 1.5531)
*   **Variable 14:** Rank 37 (Residual: 1.6620)

---

## Top-10 Diagnostic Ranking Breakdown
This table matches the data points visualized within `plot_topk.pdf`:

| Rank | Variable Index | Anomaly Score (Z-Score) | Is Ground Truth? |
| :--- | :------------- | :---------------------- | :--------------- |
| 1 | 31 | 2.5905 | No |
| 2 | 0 | 2.2441 | Yes (★) |
| 3 | 8 | 1.0909 | Yes (★) |
| 4 | 35 | 0.9210 | No |
| 5 | 1 | 0.7213 | No |
| 6 | 18 | 0.5852 | No |
| 7 | 19 | 0.5608 | No |
| 8 | 12 | 0.5561 | Yes (★) |
| 9 | 34 | 0.5158 | No |
| 10 | 20 | 0.5073 | No |

---

## Signal Metrics Summary
*   **Raw Residual Max Magnitude:** 4.6731 (at Index 9)
*   **Raw Residual Mean:** 0.1569
*   **Raw Residual Standard Deviation:** 0.9279
*   **Anomaly Score Max (Z-Score):** 2.5905
