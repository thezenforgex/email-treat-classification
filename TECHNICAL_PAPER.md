# Technical Paper: End-to-End Email Threat Classification Using Foundational Machine Learning Algorithms

**Author:** Atanu Das  
**Institution:** Learn Depth Academy LLP - Track 1 Capstone  
**Domain:** Cybersecurity / Threat Analytics  
**Task:** Binary Supervised Classification  

---

## Abstract
Email remains the primary initial access vector for modern cyberattacks, accounting for over 90% of organizational security breaches. This paper presents an end-to-end, zero-cost machine learning framework for classifying email metadata into **Routine (Ham)** and **Potentially Threatening (Phishing/Malicious)** categories. Utilizing an empirical dataset of 3,500 email observations characterized by key observable metadata (message length, URL density, attachment frequency, high-risk keyword count, sender domain age, SPF/DKIM authentication, and TLD reputation), we systematically clean, scale, and evaluate three foundational algorithms—Logistic Regression, K-Nearest Neighbors (KNN), and Decision Trees—against a Random Forest ensemble benchmark. Our empirical results show that a tuned Logistic Regression model achieves an outstanding F1-Score of **0.9919**, Accuracy of **99.43%**, and ROC-AUC of **0.9998**, outperforming non-linear baselines while preserving complete mathematical interpretability and sub-millisecond inference latency. We demonstrate the practical utility of our approach via a lightweight, interactive Streamlit deployment.

---

## 1. Introduction
Phishing, business email compromise (BEC), and credential harvesting pose severe financial and reputational risks to organizations globally. Traditional email security gateways rely heavily on static signature matching and blacklists, which frequently fail against newly registered domains, zero-day threat variants, and obfuscated messaging techniques. 

Machine Learning (ML) offers a proactive, pattern-recognition paradigm capable of inferring threat probability based on multi-dimensional signal aggregation. However, in operational SOC (Security Operations Center) environments, model explainability and low inference latency are critical requirements; security analysts must understand *why* an email was flagged without incurring high computational cost.

---

## 2. Problem Definition
Formally, given an email observation vector \( \mathbf{x}_i \in \mathbb{R}^d \) comprising $d=9$ observable metadata features:
$$\mathbf{x}_i = [x_{i,1}, x_{i,2}, \dots, x_{i,9}]$$

Our goal is to learn a predictive mapping function $f: \mathbb{R}^d \to \{0, 1\}$ that estimates the posterior class probability:
$$P(Y_i = 1 \mid \mathbf{x}_i)$$

Where:
- $Y_i = 0$: **Routine Email** (Safe, operational message)
- $Y_i = 1$: **Potentially Threatening Email** (Phishing, malware delivery, or scam)

---

## 3. Related Work
1. **Content-Based Filtering (NLP):** Traditional approaches utilize Term Frequency-Inverse Document Frequency (TF-IDF) or Word Embeddings (BERT/RoBERTa) on raw email bodies. While highly accurate, deep learning methods incur heavy memory overhead, require GPU acceleration, and act as opaque black boxes.
2. **Metadata & Header Analysis:** Recent research highlights that email headers, domain age, authentication checks (SPF, DKIM, DMARC), and keyword density provide strong statistical signals for threat detection without parsing full sensitive email text.
3. **Foundational Classifiers:** Foundational models such as Logistic Regression and Decision Trees offer high transparency, deterministic decision boundaries, and zero hardware deployment overhead.

---

## 4. Dataset Description
The research dataset consists of **3,500 email observations** synthesized according to realistic cybersecurity threat distributions. 

### Feature Dictionary
| Feature | Type | Description |
| :--- | :--- | :--- |
| `message_length` | Continuous | Total body character count |
| `url_count` | Discrete | Total count of embedded hyperlinks |
| `attachment_count` | Discrete | Count of attached files |
| `keyword_indicators` | Discrete | Count of urgent/phishing trigger keywords |
| `domain_age_days` | Continuous | Age of sending domain in days |
| `suspicious_tld` | Binary | 1 if domain TLD is high-risk (.xyz, .top, .work, .ru); 0 otherwise |
| `spf_dkim_passed` | Binary | 1 if email passed SPF/DKIM verification; 0 if failed/none |
| `uppercase_ratio` | Continuous | Proportion of uppercase characters in subject/body |
| `has_external_links` | Binary | Derived indicator (1 if `url_count` > 0; 0 otherwise) |
| `is_threat` | Binary Target | 0 = Routine, 1 = Threat |

---

## 5. Methodology & Preprocessing Pipeline
1. **Data Cleaning:** Missing values (~1.5% in `message_length` and `domain_age_days`) were imputed using feature medians to maintain robustness against skewed distributions.
2. **Feature Scaling:** Continuous variables were standard-scaled ($\mu = 0, \sigma = 1$) for distance and gradient-based algorithms:
   $$z = \frac{x - \mu}{\sigma}$$
3. **Dataset Splitting:** Stratified 80/20 train-test split (2,800 training samples, 700 evaluation samples).
4. **Cross-Validation:** 5-Fold Stratified Cross-Validation was applied during hyperparameter grid search optimization.

---

## 6. Exploratory Data Analysis (EDA)
- **Target Distribution:** ~64.7% Routine (Ham) emails vs. ~35.3% Threat emails.
- **Key Findings:**
  - Threat emails exhibited significantly lower domain ages (median < 150 days) compared to routine emails (median > 1,800 days).
  - High correlation observed between `keyword_indicators`, `suspicious_tld`, `spf_dkim_passed` failure, and the target variable `is_threat`.

---

## 7. Model Development & Hyperparameter Tuning
Four algorithms were evaluated:
1. **Logistic Regression:** Tuned with $C \in \{0.01, 0.1, 1.0, 10.0\}$ using `liblinear` and `lbfgs` solvers.
2. **K-Nearest Neighbors (KNN):** Tuned with $k \in \{3, 5, 7, 9, 11\}$ and `uniform` vs. `distance` weighting.
3. **Decision Tree Classifier:** Tuned with `max_depth` $\in \{3, 5, 7, 10, \text{None}\}$ and Gini / Entropy criteria.
4. **Random Forest (Ensemble Benchmark):** Evaluated with 100 trees for performance upper-bound reference.

---

## 8. Results & Performance Comparison

### Empirical Metric Summary Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Selected)** | **0.9943** | **0.9959** | **0.9879** | **0.9919** | **0.9998** |
| K-Nearest Neighbors (k=7) | 0.9914 | 0.9918 | 0.9838 | 0.9878 | 0.9977 |
| Decision Tree (depth=10) | 0.9871 | 0.9837 | 0.9798 | 0.9817 | 0.9855 |
| *Random Forest (Benchmark)* | *0.9943* | *0.9959* | *0.9879* | *0.9919* | *0.9990* |

---

## 9. Discussion & Error Analysis
- **Model Selection Justification:** Logistic Regression matched the ensemble benchmark in F1-score (0.9919) while offering complete parameter transparency.
- **False Negative Analysis:** Minimizing False Negatives (flagging a threat as routine) is vital. Logistic Regression achieved a Recall of **98.79%**, missing fewer than 1.2% of malicious emails.
- **Feature Weights:** `spf_dkim_passed` (negative weight) and `domain_age_days` (negative weight) alongside `keyword_indicators` (positive weight) proved to be the strongest decision drivers.

---

## 10. Limitations & Future Scope
- **Limitations:** Metadata-only models can be susceptible to adversarial sender domain spoofing if SPF/DKIM checks are bypassed via compromised legitimate infrastructure.
- **Future Scope:** Integrating lightweight TF-IDF n-gram features from subject lines and deploying automated rule-updating loops via active learning.

---

## 11. Conclusion
This study successfully demonstrates an end-to-end, reproducible email threat classification prototype adhering to zero-cost constraints. Logistic Regression provides the optimal balance between high classification performance (99.43% accuracy, 0.9919 F1-score), interpretability, and execution speed.

---

## 12. References
1. Sahingoz, O. K., et al. (2019). "Machine learning based phishing detection system." *Expert Systems with Applications*.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
3. RFC 7208: Sender Policy Framework (SPF) for Authorizing Use of Domains in Email.
