try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
except ImportError as e:
    print(f"Missing package: {e}")
    exit(1)

# =====================================================
# 1. Load Dataset
# =====================================================

df = pd.read_csv("../data/ai4i2020.csv")
df.columns = df.columns.str.strip()

# =====================================================
# 2. Feature Engineering
# =====================================================

df = df.drop(["UDI", "Product ID"], axis=1, errors="ignore")
df = df.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1, errors="ignore")

df = pd.get_dummies(df, columns=["Type"], drop_first=True)

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

joblib.dump(X.columns.tolist(), "../models/feature_columns.pkl")

print("\nOriginal Target Distribution:")
print(y.value_counts())

# =====================================================
# 3. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 4. Logistic Regression (Baseline)
# =====================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)

log_train_pred = log_model.predict(X_train_scaled)
log_test_pred = log_model.predict(X_test_scaled)

print("\n==============================")
print("Logistic Regression")
print("==============================")

print("Train Accuracy:", accuracy_score(y_train, log_train_pred))
print("Test Accuracy :", accuracy_score(y_test, log_test_pred))
print("ROC-AUC       :", roc_auc_score(y_test,
                                      log_model.predict_proba(X_test_scaled)[:,1]))
print(classification_report(y_test, log_test_pred))

# =====================================================
# 5. Random Forest (Baseline)
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

print("\n==============================")
print("Random Forest (Baseline)")
print("==============================")

print("Train Accuracy:", accuracy_score(y_train, rf_train_pred))
print("Test Accuracy :", accuracy_score(y_test, rf_test_pred))
print("ROC-AUC       :", roc_auc_score(y_test,
                                      rf_model.predict_proba(X_test)[:,1]))
print(classification_report(y_test, rf_test_pred))

# =====================================================
# 6. Random Forest + SMOTE
# =====================================================

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Training Distribution:")
print(pd.Series(y_train_smote).value_counts())

rf_smote = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_smote.fit(X_train_smote, y_train_smote)

rf_smote_train_pred = rf_smote.predict(X_train_smote)
rf_smote_test_pred = rf_smote.predict(X_test)

print("\n==============================")
print("Random Forest + SMOTE")
print("==============================")

print("Train Accuracy:", accuracy_score(y_train_smote, rf_smote_train_pred))
print("Test Accuracy :", accuracy_score(y_test, rf_smote_test_pred))
print("ROC-AUC       :", roc_auc_score(y_test,
                                      rf_smote.predict_proba(X_test)[:,1]))
print(classification_report(y_test, rf_smote_test_pred))

# =====================================================
# 7. Overfitting Analysis
# =====================================================

print("\n==============================")
print("Overfitting Analysis")
print("==============================")

print("Logistic Gap:",
      accuracy_score(y_train, log_train_pred) -
      accuracy_score(y_test, log_test_pred))

print("RF Baseline Gap:",
      accuracy_score(y_train, rf_train_pred) -
      accuracy_score(y_test, rf_test_pred))

print("RF + SMOTE Gap:",
      accuracy_score(y_train_smote, rf_smote_train_pred) -
      accuracy_score(y_test, rf_smote_test_pred))

# =====================================================
# 8. Real Prediction Output (Readable)
# =====================================================

def convert_prediction(value):
    if value == 1:
        return "Machine Will Fail"
    else:
        return "Machine Operating Normally"

sample_X = X_test.iloc[:10]
sample_preds = rf_smote.predict(sample_X)
sample_probs = rf_smote.predict_proba(sample_X)[:,1]

results_df = sample_X.copy()
results_df["Prediction"] = [convert_prediction(p) for p in sample_preds]
results_df["Failure Probability"] = sample_probs

print("\nSample Real Predictions (RF + SMOTE):")
print(results_df[["Prediction", "Failure Probability"]])

# =====================================================
# 9. Save Best Model
# =====================================================

joblib.dump(rf_smote, "../models/best_model.pkl")
print("\nBest Model (RF + SMOTE) saved successfully.")