# 🔧 Equipment Failure Prediction System

A Machine Learning project that predicts whether industrial equipment will fail based on operational parameters.

This project covers end-to-end ML workflow including:
- Data Preprocessing
- Class Imbalance Handling (SMOTE)
- Model Training & Comparison
- Model Evaluation
- Streamlit Web App Deployment

---

## 📌 Problem Statement

Unexpected machine failures can cause:

- Production downtime
- Increased maintenance cost
- Safety risks
- Revenue loss

This project builds a classification model to predict machine failure in advance, enabling predictive maintenance.

---

## 📊 Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

Target Variable:
- `0` → Machine Operating Normally
- `1` → Machine Will Fail

Features Used:
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)
- Machine Type

---

## ⚙️ Data Preprocessing

- Checked missing values
- Removed duplicates
- Label encoding for Machine Type
- Feature Scaling
- Handled class imbalance using **SMOTE**

After SMOTE distribution:

0 → 7729  
1 → 7729  

Balanced dataset for training.

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression

- Test Accuracy: **82.45%**
- ROC-AUC: **0.9069**

Observation:
Logistic Regression struggled with minority class precision.

---

### 2️⃣ Random Forest (Baseline)

- Test Accuracy: **98.05%**
- ROC-AUC: **0.9619**

Observation:
Very strong performance and high precision.

---

### 3️⃣ Random Forest + SMOTE

- Test Accuracy: **96.3%**
- ROC-AUC: **0.9584**

Observation:
Improved recall for failure class after balancing data.

---

## 🏆 Final Model Selection

Random Forest was selected due to:

- High accuracy
- Strong ROC-AUC score
- Better balance between precision and recall

---

## 📈 Model Evaluation Metrics

Metrics Used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Confusion matrix and classification report were analyzed for model comparison.

---

## 🖥️ Streamlit Web Application

An interactive web app was built using Streamlit.

### Features:
- User input form
- Real-time prediction
- Failure probability display
- Clear visual result (Fail / Normal)

To run locally:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

---

## 📂 Project Structure

```
Equipment-Failure-Prediction/
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│
├── outputs/
│
├── src/
│   ├── train.py
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit

---

## 💼 Business Impact

This system helps industries:

- Reduce unexpected breakdowns
- Improve maintenance planning
- Lower operational costs
- Increase equipment reliability

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Feature importance visualization in app
- Cloud deployment (AWS / Render / Streamlit Cloud)

---

## 👩‍💻 Author

Akshata Hipparkar

If you found this project useful, please ⭐ the repository.
