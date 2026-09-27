# 🛡️ Email Threat Classification — Zero-Cost ML Capstone Project

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-ff4b4b.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Foundational%20ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, zero-cost machine learning capstone project for **Email Threat Classification**. This repository contains complete research, exploratory data analysis (EDA), model training, evaluation metrics, interactive web application deployment, formal technical paper, and presentation slides.

---

## 📂 Repository Structure

```
.
├── .streamlit/
│   └── config.toml               # Streamlit theme & UI configuration
├── data/
│   ├── raw_email_threat_dataset.csv        # Raw dataset with missing values & noise
│   └── processed_email_threat_dataset.csv  # Cleaned, imputed, and scaled dataset
├── figures/
│   ├── class_distribution.png              # Class balance & EDA boxplots
│   ├── confusion_matrices.png              # Comparative confusion matrices
│   ├── correlation_matrix.png              # Feature correlation matrix heatmap
│   ├── feature_importance.png              # Model feature weight breakdown
│   └── roc_curves.png                      # Comparative ROC curves
├── models/
│   ├── best_model.joblib                   # Selected foundational model (Logistic Regression)
│   ├── model_comparison_results.csv        # Benchmarking matrix across all models
│   ├── model_metadata.json                 # Hyperparameters & evaluation metrics
│   └── scaler.joblib                       # Fitted StandardScaler pipeline
├── src/
│   ├── __init__.py                         # Python package initializer
│   ├── data_preprocessing.py               # Preprocessing & cleaning pipeline
│   ├── generate_dataset.py                 # Dataset generator script
│   ├── model_training.py                   # Model training, grid search & evaluation
│   └── predict.py                          # Real-time & batch inference pipeline
├── .gitignore                              # Git ignore file
├── DEPLOYMENT_GUIDE.md                     # Step-by-step GitHub & Streamlit deployment guide
├── PRESENTATION.md                         # Presentation slide deck & viva defense guide
├── README.md                               # Project documentation (this file)
├── TECHNICAL_PAPER.md                      # Formal Capstone Technical Research Paper
├── app.py                                  # Interactive Streamlit Web Application
├── email_threat_classification.ipynb       # Fully executed end-to-end Jupyter Notebook
└── requirements.txt                        # Python package dependencies
```

---

## ⚡ Quick Start & Reproduction Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/email-threat-classification.git
cd email-threat-classification
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Model Training
Generate the raw dataset, run EDA preprocessing, grid search hyperparameter tuning, and evaluate foundational models:
```bash
python src/generate_dataset.py
python src/model_training.py
```

### 3. Launch Interactive Streamlit Web App
Launch the web dashboard locally to perform real-time predictions and batch CSV scans:
```bash
streamlit run app.py
```

---

## 🏆 Model Benchmarking Summary

Evaluating 5-fold cross-validated performance on 700 unseen test observations:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Selected)** | **99.43%** | **99.59%** | **98.79%** | **0.9919** | **0.9998** | 🥇 **Best Foundational** |
| K-Nearest Neighbors ($k=7$) | 99.14% | 99.18% | 98.38% | 0.9878 | 0.9977 | 🥈 High Accuracy |
| Decision Tree (Depth=10) | 98.71% | 98.37% | 97.98% | 0.9817 | 0.9855 | 🥉 High Explainability |
| *Random Forest (Benchmark)* | *99.43%* | *99.59%* | *98.79%* | *0.9919* | *0.9990* | 🔍 Ensemble Baseline |

---

## 📖 Key Project Deliverables
- **Interactive Web App:** [`app.py`](./app.py)
- **Deployment Guide:** [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md)
- **Jupyter Notebook:** [`email_threat_classification.ipynb`](./email_threat_classification.ipynb)
- **Technical Research Paper:** [`TECHNICAL_PAPER.md`](./TECHNICAL_PAPER.md)
- **Presentation Deck & Viva Prep:** [`PRESENTATION.md`](./PRESENTATION.md)

---
*Developed for Learn Depth Academy LLP - Track 1 Capstone Project (Problem 06).*
