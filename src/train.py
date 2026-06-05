import pandas as pd

from data_processing import FeatureEngineer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =========================
# 1. LOAD RAW DATA
# =========================
df_raw = pd.read_csv("data/raw/data.csv")

# =========================
# 2. FEATURE ENGINEERING
# =========================
fe = FeatureEngineer()
df = fe.fit_transform(df_raw)

print("Data preview:")
print(df.head())

# =========================
# 3. PROXY TARGET CREATION
# =========================
df['is_high_risk'] = (
    df['recency'] > df['recency'].quantile(0.7)
).astype(int)

print("\nClass distribution:")
print(df['is_high_risk'].value_counts())

# =========================
# 4. FEATURES / TARGET SPLIT
# =========================
X = df.drop(columns=["is_high_risk", "CustomerId"])
y = df["is_high_risk"]

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 7. MODEL (FIXED FOR IMBALANCE)
# =========================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# 8. PREDICTIONS
# =========================
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# =========================
# 9. EVALUATION
# =========================
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print("F1:", f1_score(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))