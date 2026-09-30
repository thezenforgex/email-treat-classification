# 🛡️ Email Threat Classification — ML Capstone Project
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-ff4b4b.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Foundational%20ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project for **Email Threat Classification**. This repository contains the standalone interactive web application, model artifacts, notebook analysis, and project documentation.

## 🚀 Live Demo
**[https://atanudas-email-treat-classification.streamlit.app/](https://atanudas-email-treat-classification.streamlit.app/)**

---

## 📂 Repository Structure

```
.
├── data/
│   └── spam_dataset.csv               # Email Threat Dataset (1,000 records)
├── model/
│   ├── model.pkl                      # Serialized ML Classification Model
│   └── vectorizer.pkl                 # Feature Scaler / Vectorizer Artifact
├── README.md                          # Project documentation (this file)
├── app.py                             # Self-contained Streamlit Web Application
├── email_threat_classification.ipynb  # End-to-End Jupyter Notebook
└── requirements.txt                   # Python dependencies
```

---

## ⚡ Quick Start & Reproduction Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/thezenforgex/email-treat-classification.git
cd email-treat-classification
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Web App
Launch the web application locally:
```bash
streamlit run app.py
```

---

## 🏆 Model & Feature Extraction Summary

- **Classification Model:** Logistic Regression (`model/model.pkl`)
- **Vectorizer / Scaler:** Standard Scaler (`model/vectorizer.pkl`)
- **Extracted Features:** 
  - Message Length & Uppercase Character Ratio
  - High-Risk Keyword Count & Suspicious TLD Detection
  - Attachment Pattern Analysis & Link Verification
  - SPF / DKIM Authentication Check

---

## 📖 Key Project Files
- **Dataset:** [`data/spam_dataset.csv`](./data/spam_dataset.csv)
- **Interactive Web App:** [`app.py`](./app.py)
- **Model Artifacts:** [`model/`](./model/)
- **Jupyter Notebook Analysis:** [`email_threat_classification.ipynb`](./email_threat_classification.ipynb)
- **Live Demo Link:** [https://atanudas-email-treat-classification.streamlit.app/](https://atanudas-email-treat-classification.streamlit.app/)