# Loan-Status-Prediction
Predicts using features if loan should be granted or not. Explore data, visualise relationships, select relevant features and build models for evaluation. 

# 🏦 Loan Prediction Using Machine Learning

A supervised binary classification project that predicts whether a loan application will be **approved or rejected** based on applicant demographics, financial information, and credit behaviour.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Models](#models)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

Financial institutions process thousands of loan applications manually, which is time-consuming and prone to inconsistency. This project builds a machine learning pipeline to automate and support loan approval decisions using structured applicant data.

**Task:** Binary Classification — predict `Loan_Status` (Approved = 1, Rejected = 0)  
**Dataset Source:** [Analytics Vidhya – Loan Prediction III](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii)

---

## Dataset

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Applicant's gender |
| Married | Categorical | Marital status |
| Dependents | Numerical | Number of dependents (0, 1, 2, 3+) |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income of applicant |
| CoapplicantIncome | Numerical | Monthly income of co-applicant |
| LoanAmount | Numerical | Loan amount requested |
| Loan_Amount_Term | Numerical | Loan term in months |
| Credit_History | Binary | 1 = meets guidelines, 0 = otherwise |
| Property_Area | Categorical | Urban, Semiurban, Rural |
| **Loan_Status** | **Target** | **Y = Approved, N = Rejected** |

---

## Project Pipeline

```
Raw Data → EDA → Data Cleaning → Feature Engineering → Encoding & Scaling → Model Training & Tuning → Evaluation
```

### 1. Exploratory Data Analysis
- Distribution plots for all numerical features (income, loan amount, credit history)
- Categorical feature breakdown vs. loan status using percentage bar plots
- Correlation heatmap to identify relationships between variables

### 2. Data Cleaning
- Dropped `Loan_ID` (non-informative identifier)
- Replaced `3+` in `Dependents` with `4` for numeric conversion
- Imputed missing numerical values using **median**
- Imputed missing categorical values using **mode**

### 3. Feature Engineering
- **TotalIncome** = `ApplicantIncome` + `CoapplicantIncome`
- **TotalIncome_log** = log-transformed total income to reduce right skew
- **Loan_to_Income** = `LoanAmount / TotalIncome` (affordability ratio)
- Dropped original income columns after aggregation

### 4. Encoding & Scaling
- **One-Hot Encoding** for nominal categorical variables (Gender, Married, Education, Self_Employed)
- **Ordinal Encoding** for `Property_Area` (Rural < Urban < Semiurban), assuming an ordinal relationship
- **StandardScaler** applied within pipelines for Logistic Regression and Naive Bayes

---

## Models

All three models were trained using **sklearn Pipelines** with **Sequential Forward Feature Selection** and **hyperparameter tuning via GridSearchCV / RandomizedSearchCV** with Stratified K-Fold cross-validation.

### Naive Bayes (GaussianNB)
- **Selected Features:** Credit_History, TotalIncome_log, Property_Area_Ord, Married_Yes, Education_Not Graduate
- **Best var_smoothing:** `1e-12`

### Logistic Regression
- **Selected Features:** LoanAmount, Credit_History, Gender_Male, Married_Yes, Education_Not Graduate
- **Best C:** `0.1` (L2 regularisation)
- **Best CV Accuracy:** 79.8%

### Random Forest
- **Selected Features:** Credit_History, Property_Area, Gender, Married, Self_Employed
- **Best Params:** `n_estimators=100`, `max_depth=None`, `min_samples_leaf=2`, `min_samples_split=2`
- **Best CV AUC:** 74.7%

---

## Results

Performance evaluated at multiple probability thresholds (0.3, 0.5, 0.8) to explore the precision–recall trade-off.

| Model | Accuracy (0.5 threshold) | AUC |
|---|---|---|
| Naive Bayes | ~85% | Competitive |
| Logistic Regression | ~85% | Competitive |
| Random Forest | ~85% | **Highest** |

> **Random Forest** achieved the highest AUC score and demonstrated the most **balanced** classification between approved and rejected cases, making it slightly better at identifying risky applicants compared to the linear models.

### Threshold Analysis
| Threshold | Effect |
|---|---|
| 0.3 | High recall (0.99) — minimises missed approvals, increases false positives |
| 0.5 | Balanced default behaviour |
| 0.8 | High precision — reduces risky approvals, increases missed eligible applicants |

---

## Key Findings

- **Credit History** is the single most dominant predictor: ~80% approval rate with good credit history vs. ~90% rejection rate without it
- **Property Area** is the second most important feature — Semiurban applicants had the highest approval rates (~75–80%)
- **Income skewness** required log transformation; combining applicant and co-applicant income into `TotalIncome` improved signal quality
- All three models performed similarly, suggesting the dataset has a few dominant features — more diverse data would likely yield greater differentiation between model complexities

---

## Technologies Used

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — preprocessing, modelling, evaluation
  - `LogisticRegression`, `GaussianNB`, `RandomForestClassifier`
  - `Pipeline`, `GridSearchCV`, `RandomizedSearchCV`
  - `SequentialFeatureSelector`, `StratifiedKFold`
  - `StandardScaler`, `SimpleImputer`, `OneHotEncoder`
  - `roc_auc_score`, `classification_report`, `confusion_matrix`

---

## Future Work

- Incorporate additional financial indicators (debt-to-income ratio, employment tenure)
- Apply **cost-sensitive learning** to reflect real-world asymmetric lending risks
- Investigate **fairness / bias** considerations across demographic groups
- Expand dataset size and explore temporal / behavioural features
- Test gradient boosting models (XGBoost, LightGBM) for potential accuracy gains

---

## Team

| Name | Student ID |
|---|---|
| Imaan Shahid | 483966 |
| Sushreeta Pal | 478579 |
| Anuja Limaye | 482706 |
| Neha Barhanpurkar | 481284 |
