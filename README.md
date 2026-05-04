# 🫀 CHD Fuzzy Expert System

A Python-based expert system for predicting Coronary Heart Disease (CHD) using fuzzy logic techniques.

---

## 🚀 Overview

This project implements a **Fuzzy Inference System (FIS)** to evaluate heart disease risk based on:

- Blood Pressure (BP)
- Cholesterol Level (Chol)
- Heart Rate (HR)

The system uses both:
- Mamdani inference (Centroid)
- Sugeno inference

---

## 🧠 Features

- Fuzzification of medical inputs
- Rule-based inference system
- Mamdani + Sugeno outputs
- Hedge-modified fuzzy logic (Indeed / Somewhat)
- Sensitivity analysis
- Visualization of:
  - Membership functions
  - Output aggregation
  - 3D surface
- CSV export of results

---

## 🧪 Sample Outputs

### 🔹 Membership Functions
![Membership](ExpertSystemCHD\Outputs\membership_bp.png)

### 🔹 Aggregated Output
![Output](images/person_1_aggregated_base.png)

### 🔹 Sensitivity Analysis
![Sensitivity](images/sensitivity_analysis.png)

---

## 🧱 System Components

- Membership Functions:
  - Triangular (trimf)
  - Trapezoidal (trapmf)
- Rule Evaluation (min operator)
- Aggregation (max operator)
- Defuzzification:
  - Centroid (COG)
  - Sugeno weighted average

---

## 🛠️ Technologies

- Python 3
- NumPy
- Pandas
- Matplotlib

---

## ▶️ How to Run

```bash
HealthWise.py
```

---

## 📊 Output

The system generates:
- CSV files (results, rules, fuzzification)
- PNG plots (graphs)
- Execution summary (text file)

---

## 📌 Notes

- Uses linguistic labels: Healthy / Middle / Sick
- Includes hedge logic:
  - Indeed Healthy (μ²)
  - Somewhat Sick (√μ)

---

## 👨‍💻 Author

Fawaz Mohammed Alwahidi
