# 🔧 Equipment Failure Prediction System
### Traditional Machine Learning Approach

An end-to-end Machine Learning project that predicts whether industrial equipment will fail based on operational sensor data.

This project covers:
- Data Cleaning & Preprocessing
- Feature Engineering
- Class Imbalance Handling (SMOTE)
- Model Training & Comparison
- Overfitting Analysis
- Model Saving
- Streamlit Web App Deployment

---

## 📌 Objective

Build a binary classification model to predict equipment failure using operational sensor parameters and compare multiple traditional ML algorithms to select the best-performing model.

---

## 📊 Dataset

Dataset Used:  
**AI4I 2020 Predictive Maintenance Dataset**

Source:  
https://www.kaggle.com/datasets/geetanjalisikarwar/equipment-failure-prediction-dataset  

### 🎯 Target Variable
Machine failure  
- `0` → Machine Operating Normally  
- `1` → Machine Will Fail  

### 📈 Features Used
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)
- Machine Type

---

## ⚙️ Project Workflow

1. Data Loading  
2. Data Cleaning  
3. Feature Engineering  
4. Stratified Train-Test Split  
5. Feature Scaling (for Logistic Regression)  
6. Handling Class Imbalance  
   - `class_weight="balanced"`
   - SMOTE Oversampling  
7. Model Training  
8. Model Evaluation  
9. Overfitting Analysis  
10. Model Saving  
11. Streamlit Deployment  

---

## 🧹 Data Preprocessing & Feature Engineering

- Removed unnecessary identifiers (`UDI`, `Product ID`)
- Removed individual failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
- One-hot encoded Machine Type
- Applied StandardScaler (Logistic Regression)
- Stratified split to preserve class distribution
- Handled class imbalance using SMOTE

### After SMOTE Distribution:
0 → 7729  
1 → 7729  

Balanced training dataset.

---

## 🤖 Models Implemented & Results

### 1️⃣ Logistic Regression

- Test Accuracy: **82.45%**
- ROC-AUC: **0.9069**

Observation:  
Struggled with minority class precision despite balanced weighting.

---

### 2️⃣ Random Forest (Baseline)

- Test Accuracy: **98.05%**
- ROC-AUC: **0.9619**

Observation:  
Very strong overall performance but slight class imbalance effect.

---

### 3️⃣ Random Forest + SMOTE (Final Model)

- Test Accuracy: **96.3%**
- ROC-AUC: **0.9584**

Observation:  
Improved recall for failure class and better balance between precision and recall.

---

## 📈 Evaluation Metrics Used

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## 🔎 Overfitting Analysis

Train-Test accuracy gap was analyzed for all models.

Observation:
- Random Forest Baseline showed higher training accuracy compared to testing accuracy.
- SMOTE improved minority class recall.
- Final model chosen based on better class balance rather than only highest accuracy.

---

## 🏆 Final Model Selection

**Random Forest + SMOTE** was selected as the final model due to:

- Strong ROC-AUC
- Improved recall on failure class
- Balanced overall performance
- Better real-world applicability

The trained model is saved as:

```
models/best_model.pkl
```

---

## 🖥️ Streamlit Web Application

An interactive prediction interface was built using Streamlit.

### Features:
- User input form
- Real-time failure prediction
- Failure probability display
- Clear visual output (Fail / Normal)

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-link>
cd Equipment-Failure-Prediction
```

---

### 2️⃣ Create Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train the Model

Place dataset inside:

```
data/ai4i2020.csv
```

Then run:

```bash
python train.py
```

This will:
- Train all models
- Perform evaluation
- Save the best model in `models/`

---

### 5️⃣ Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 📂 Repository Structure

```
Equipment-Failure-Prediction/
│
├── data/                → Dataset
├── models/              → Saved model
├── outputs/             → Evaluation outputs
├── src/
│   ├── train.py         → Model training
│   └── app.py           → Streamlit app
│
├── requirements.txt
├── README.md
└── ASSUMPTIONS.md
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

This system enables:

- Predictive maintenance
- Reduced unexpected breakdowns
- Lower operational costs
- Improved equipment reliability

---

## 👩‍💻 Author

Akshata Hipparkar  

If you found this project useful, please ⭐ the repository.
