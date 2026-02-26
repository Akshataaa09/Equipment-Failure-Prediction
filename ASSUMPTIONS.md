# Assumptions, Trade-offs, and Improvements

## Assumptions Made

1. Sensor readings are accurate and properly calibrated.
2. Machine failure label is correctly annotated.
3. Each row is treated as an independent observation.
4. Temporal dependency was not modeled.
5. Class imbalance exists and requires handling.

---

## Trade-offs Considered

### Logistic Regression

Pros:
- Interpretable
- Fast training
- Stable generalization

Cons:
- Cannot capture nonlinear patterns

### Random Forest

Pros:
- Captures nonlinear relationships
- Handles feature interactions
- Robust to noise

Cons:
- Less interpretable
- Higher computational cost

### SMOTE

Pros:
- Improves minority class recall
- Balances dataset effectively

Cons:
- Synthetic data may introduce noise
- Can increase overfitting risk

Trade-off:
Performance vs interpretability.
Random Forest + SMOTE chosen for higher predictive power.

---

## Overfitting Observations

- Logistic Regression showed low variance.
- Random Forest baseline showed higher training accuracy.
- RF + SMOTE slightly increased variance but improved recall.

Overfitting was monitored using:
Train-Test accuracy gap comparison.

---

## Production Improvements

If deployed in real industrial setting:

1. Use Cross-Validation instead of single split.
2. Perform Hyperparameter tuning (GridSearchCV).
3. Add time-series feature engineering.
4. Use SHAP for model explainability.
5. Implement monitoring for data drift.
6. Build CI/CD pipeline for automated retraining.
7. Add threshold tuning based on business cost.

---

## Real-World Consideration

In industrial systems:
False negatives (missed failure) are more costly than false positives.
Model threshold can be adjusted accordingly.