# 💧 HydroScan AI: Clandestine Bypass Detection via Hydraulic Transients

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mechatronics](https://img.shields.io/badge/Domain-Mechatronics%20Engineering-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Executive Overview

**HydroScan AI** is an advanced diagnostic framework designed to tackle **Non-Revenue Water (NRW)** losses. By leveraging **Digital Signal Processing (DSP)** and **Supervised Machine Learning**, this system identifies illicit water derivations (bypasses) through the analysis of **Hydraulic Transients** (Water Hammer events). 

Unlike traditional methods, this project utilizes high-frequency pressure telemetry to detect micro-anomalies in wave propagation, classified through optimized **Support Vector Machines (SVM)** and **Random Forest (RF)** architectures.

---

## 🚀 Key Technical Features

* **Adaptive Data Augmentation:** A built-in synthetic generator that mitigates data scarcity by applying Gaussian noise injection, temporal warping, and amplitude scaling to real-world signals.
* **Multi-Domain Feature Extraction:**
    * **Time-Domain:** Non-linear curve fitting for exponential decay constant ($\tau$) estimation.
    * **Frequency-Domain:** Spectral energy distribution via Fast Fourier Transform (FFT).
    * **Time-Frequency:** Multilevel **Discrete Wavelet Transform (DWT)** using Daubechies 4 (`db4`) wavelets for transient localization.
* **Industrial-Grade GUI:** A PyQt5-based interface for real-time signal visualization and model performance benchmarking.
* **Production-Ready Inference:** Seamless loading of serialized `joblib` models for binary classification of pipeline integrity.

---

## 🔬 Methodology & Signal Analysis

The core of the detection logic lies in the physics of transient waves. When a bypass is present, the pressure wave reflection pattern changes. 

### Feature Engineering Pipeline
1.  **Baseline Extraction:** Median filtering for nominal pressure identification.
2.  **Transient Characterization:** Detection of peak amplitude and time-to-peak.
3.  **Wavelet Decomposition:** The signal is decomposed into 4 levels to isolate high-frequency noise from the hydraulic structural response.



---

## 📂 Project Structure

```text
├── models/               # Serialized .joblib models & Scalers
├── data/                 # Raw and augmented telemetry (CSV)
├── src/
│   ├── train_real_gui.py # Advanced Training & Augmentation Suite
│   ├── inference_eng.py  # Real-time Diagnostic Engine
│   └── dsp_utils.py      # Core Signal Processing algorithms
├── requirements.txt      # Environment dependencies
└── README.md             # Project documentation
