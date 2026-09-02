## Forked from AERCA: Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery (ICLR 2025 Oral)

[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/theamrzaki/AERCA/blob/main/LICENSE)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

# SFlexRCA: Lightweight, Scalable, and Flexible Root Cause Analysis for IIoT Edge Systems


---

## 🗂️ Table of Contents

1. [Overview](#overview)
2. [System Configuration](#system-configuration)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Datasets](#datasets)

---

## 📘 Overview

The SFlexRCA is a lightweight, scalable, and flexible root cause analysis framework designed for Industrial Internet of Things (IIoT) edge systems. It addresses the challenges of high-dimensional sensor telemetry and fault propagation in interconnected components by transforming multivariate telemetry into compact orthogonal representations and applying shared lightweight linear modeling. This approach avoids the need for explicit graph construction, message passing, and per-variable or lag-specific parameter growth. 

SFlexRCA has been evaluated on three publicly available IIoT datasets (BATADAL, SWAT, and WADI) and has demonstrated superior performance in terms of RCA accuracy, training and inference time, training memory, and edge energy consumption compared to various statistical, causal, and non-causal baselines. The framework has also been deployed on Raspberry Pi 3 and Raspberry Pi 5 to assess its practical suitability for resource-constrained IIoT edge environments.

---

## 🧰 System Configuration

All experiments are conducted using multiple random seeds on a Linux workstation equipped with an Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (20 cores), 32~GB RAM, and an NVIDIA GeForce RTX 3070 GPU with 8~GB memory, running Ubuntu 22.04.2 LTS. The implementation uses Python 3.10.12, PyTorch 2.7.1+cu126 with CUDA 12.6, and PyTorch Geometric 2.6.1.

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10:** Ensure that Python 3.10 is installed.
- **Virtual Environment (Recommended):** It is advisable to use a virtual environment to manage dependencies.

### Steps

1. **Install `virtualenv` (if not already installed):**

   ```bash
   python3 -m pip install --user virtualenv
    ```
   
2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   ```
3. **Activate the virtual environment:**

   ```bash
    source venv/bin/activate
    ```
4. **Install the required packages:**
    
    ```bash
    pip install -r requirements.txt
    ```
   
5. **Deactivate the virtual environment (when done):**

   ```bash
   deactivate
   ```
---

## 🚀 Usage

### To Replicate RQ1/RQ2/RQ3To Replicate RQ1/RQ2/RQ3

```bash
    ./RQ_1.sh
    ./RQ_3_ablations_search.sh
    ./RQ_4_ablations_search_windows.sh
    ./RQ_5_ablations_search_projections.sh
```

Two additional branches dedicated to RQ2 to run on Raspberry Pi 3 and Raspberry Pi 5 are available in the repository. 
To replicate RQ2, please refer to the respective branches.
- (from-raspberrypi-larger)
- (from-raspberrypi-smaller)



---

## 📊 Datasets
The repository includes support for multiple datasets, each designed to evaluate the algorithm under different conditions:

- [BATADAL](https://github.com/hanxiao0607/AERCA/tree/main/datasets/batadal): A dataset for anomaly detection for water distribution systems.
- [SWaT](https://github.com/hanxiao0607/AERCA/tree/main/datasets/swat): A dataset for anomaly detection in water treatment systems.
- [WADI](https://github.com/hanxiao0607/AERCA/tree/main/datasets/wadi): A dataset for anomaly detection for water distribution systems.

Ensure that the dataset you choose is formatted as expected by the code. Additional preprocessing scripts or instructions may be provided within the repository as needed.

---
