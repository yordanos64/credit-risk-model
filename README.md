# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to measure, monitor, and manage credit risk in a transparent and reliable manner. Therefore, credit scoring models should be interpretable so that banks can explain how lending decisions are made to regulators, auditors, management, and customers. A well-documented model provides clear records of data sources, feature engineering steps, modeling assumptions, and evaluation procedures. This improves regulatory compliance, supports model validation, and ensures that lending decisions can be justified and audited. As a result, interpretability and documentation are critical requirements when developing credit risk models in regulated banking environments.

## 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

The dataset used in this project does not contain a direct default indicator that identifies whether a customer failed to repay a loan. Because supervised machine learning models require a target variable, a proxy variable must be created. In this project, customer behavior is analyzed using Recency, Frequency, and Monetary (RFM) metrics, and customers are segmented using clustering techniques. Customers with low engagement and low transaction activity are treated as high-risk customers and assigned a proxy risk label.

However, proxy variables are not true observations of default behavior. As a result, the generated labels may not perfectly represent actual credit risk. This can introduce misclassification errors, bias, and inaccurate risk estimates. Therefore, the proxy target should be considered a modeling assumption rather than ground truth, and its limitations should be clearly documented.

## 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) transformations are highly interpretable. They allow risk analysts and regulators to understand how individual variables influence predictions, making them easier to validate, monitor, and explain. However, these models may not always achieve the highest predictive performance.

More advanced models such as Random Forests, XGBoost, or Gradient Boosting often provide better predictive accuracy because they can capture complex relationships in the data. However, they are more difficult to interpret and explain, which can create challenges for regulatory compliance and model governance.

In regulated financial environments, organizations must balance predictive performance with transparency, explainability, and regulatory requirements. For this reason, interpretable models are often preferred when the performance difference is not significant.
# 📘 Project Progress (Task 2 → Task 5)



## 🧪 Task 2 — Exploratory Data Analysis (EDA)

### 🎯 Objective
Understand the structure, distribution, and quality of the transaction dataset.

### 🛠 Work Done
- Loaded raw transaction data (`data.csv`)
- Checked missing values and data types
- Explored distributions of transaction amounts
- Identified customer-level aggregation requirement

### 📌Key Insight
Raw transaction data is not suitable for direct modeling. It must be aggregated at customer level.

---

#Task 3 — Feature Engineering Pipeline

### 🎯 Objective
Transform raw transaction data into customer-level machine learning features.

### 🛠 Work Done
Implemented `FeatureEngineer` class:
- Converted transaction timestamps to datetime
- Created snapshot-based RFM features:
  - Recency (days since last transaction)
  - Frequency (number of transactions)
  - Monetary value (sum of transactions)
- Added statistical features:
  - mean, std, max of transaction amounts
- Built sklearn-compatible transformer pipeline

### 📌 Output Features
- recency
- frequency
- monetary_amount
- monetary_value
- amount_mean
- amount_std
- amount_max

---

## 🧠 Task 4 — Proxy Label Creation

### 🎯 Objective
Create a target variable since no real default labels exist.

### 🛠 Work Done
- Applied behavioral segmentation using RFM features
- Defined high-risk customers using recency-based thresholding
- Generated binary target variable:
  - `is_high_risk = 1` (high risk)
  - `is_high_risk = 0` (low risk)

### ⚠️ Important Note
This is a proxy label, not real credit default data.

### 📌 Key Insight
Converted an unsupervised problem into a supervised learning problem using business rules.



🤖 Task 5 — Model Training & Evaluat Objective
Build and evaluate a credit risk prediction model.

 Work Done
- Split data into training and testing sets
- Applied feature scaling using StandardScaler
- Trained Logistic Regression model
- Handled class imbalance using `class_weight="balanced"`
- Evaluated model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

 📊 Sample Performance
- Accuracy: ~0.70–0.80  
- ROC-AUC: ~0.68–0.75  

 📌 Key Insight
Model performance is influenced by proxy label construction and class imbalance, not true default behavior.



 Overall Project Insight

This project demonstrates an end-to-end credit risk modeling pipeline:

Raw Data → Feature Engineering → Proxy Labeling → Supervised Learning

Key learnings:
- Importance of RFM feature engineering
- Challenges of missing ground-truth labels
- Impact of class imbalance on model performance
- Need for interpretable models in finance



 Next Step (Task 6)

- FastAPI deployment
- Docker containerization
- CI/CD pipeline with GitHub Actions
- `/predict` API endpoint for inference