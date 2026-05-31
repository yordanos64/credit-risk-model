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
