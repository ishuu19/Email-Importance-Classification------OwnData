# 📧 Email Importance Classification using Machine Learning

This project classifies HKBU emails into different levels of importance using machine learning techniques and natural language processing. It was developed as part of a university project to explore model evaluation, text-based feature engineering, cross-validation, and model tuning.

---

## 💡 Objective

To build a machine learning pipeline that classifies emails into importance levels (1–5) based on:
- Sender
- Subject
- Email body (cleaned)

---

## 🛠️ Features & Techniques Used

- ✉️ Text Preprocessing using TF-IDF
- 🧪 Train/Test Split with Stratification
- ⚖️ SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance
- 🔁 Stratified K-Fold Cross-Validation
- 📊 Model Comparison using F1-Score Tables
- ✅ Hyperparameter Tuning with GridSearchCV
- 🚫 Avoided test data leakage
- ❌ Avoided incompatible models like Naive Bayes and inappropriate regression models

---

## 🧠 Models Evaluated

The following classifiers were trained and compared:

- Logistic Regression (multi-class)
- Random Forest
- Gradient Boosting Classifier
- Support Vector Machine (SVC)

Final Model: 📈 The Random Forest model was selected based on macro F1-score performance.

---

## 📈 Evaluation Results (Unseen Test Set)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 1.0   | 0.9569    | 0.9592 | 0.9581   |
| 2.0   | 0.9143    | 0.9447 | 0.9293   |
| 3.0   | 0.7870    | 0.7658 | 0.7763   |
| 4.0   | 0.9615    | 0.7576 | 0.8475   |
| 5.0   | 0.7901    | 0.7356 | 0.7619   |

- Accuracy: 91.00%
- Macro Avg F1-score: 85.46%

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn

