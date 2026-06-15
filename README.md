# 🤖 Ensemble Learning Project

A machine learning project that demonstrates **ensemble learning techniques** — Bagging and AdaBoost — applied to the **UCI Airfoil Self-Noise dataset**. The project compares both classifier and regressor variants of each ensemble method, implemented as a clean Python script exported from Google Colab.

---

## 📌 Overview

Ensemble learning improves predictive performance by combining multiple models rather than relying on a single learner. This project explores four ensemble approaches:

| Technique | Variant | Description |
|-----------|---------|-------------|
| **Bagging** | Classifier | Reduces variance by training multiple Decision Tree classifiers on random data subsets |
| **Bagging** | Regressor | Same approach applied to regression on continuous targets |
| **AdaBoost** | Classifier | Boosts weak classifiers sequentially, focusing on misclassified samples |
| **AdaBoost** | Regressor | Adaptive boosting applied to regression tasks |

All models are evaluated and scored on a held-out test set.

---

## 🗂️ Repository Structure

```
Essemble-Learning-Project/
│
├── essemble_learning.py    # Main Python script — data loading, EDA, ensemble model training & evaluation
├── NReport.docx            # Project report document
└── README.md               # Project documentation
```

> **Note:** The script was originally developed as a Google Colab notebook and exported as a `.py` file. It can be run directly or re-imported into Colab.

---

## 📦 Dataset

**Source:** [UCI Machine Learning Repository — Airfoil Self-Noise Dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)

**Loaded directly from URL:**
```
https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat
```

**Format:** Tab-separated values (TSV)
**Records:** 1,503 observations × 6 variables

### Features

| Column | Description |
|--------|-------------|
| `Frequency` | 🎯 **Target** — Airfoil frequency in Hertz |
| `Angle of attack` | Angle of attack in degrees |
| `Chord length` | Chord length in meters |
| `Free-stream velocity` | Free-stream velocity in m/s |
| `Suction/side` | Suction side displacement thickness in meters |
| `Scaled/sound` | Scaled sound pressure level in decibels |

> The dataset was obtained from NASA experiments on NACA 0012 airfoil blade sections at various wind tunnel speeds and angles of attack.

---

## 🔍 Project Pipeline

### 1. 📥 Data Loading
- Dataset loaded directly from UCI repository URL using Pandas (no local download needed)
- Tab-separated format parsed with custom column names

### 2. 🧹 Data Inspection & EDA
- Check for empty/null cells using `np.where` and `applymap`
- Generate descriptive statistics with `.describe()`
- **Correlation heatmap** using Seaborn to visualize feature relationships (10×10, `YlGnBu` palette)

### 3. ✂️ Train-Test Split
- 70% training / 30% testing split using `train_test_split` (random state = 0)
- Target variable: `Frequency`
- Features: all remaining columns
- One-hot encoding applied via `pd.get_dummies()`

### 4. 🤖 Ensemble Model Training & Evaluation

#### Bagging Classifier
```python
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

#### Bagging Regressor
```python
from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

#### AdaBoost Classifier
```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

#### AdaBoost Regressor
```python
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(x_train, y_train)
model.score(x_test, y_test)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Nimra064/Essemble-Learning-Project.git
cd Essemble-Learning-Project
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### 3. Run the Script

```bash
python essemble_learning.py
```

Or open in **Google Colab** directly using the original notebook:
🔗 [Open in Colab](https://colab.research.google.com/drive/1L64bjhInkBTt3IIUBfDCcotbIU-rLBNy)

---

## 📊 Models Compared

| Model | Type | Base Estimator | Strategy |
|-------|------|----------------|----------|
| BaggingClassifier | Classification | Decision Tree | Bootstrap sampling + averaging |
| BaggingRegressor | Regression | Decision Tree | Bootstrap sampling + averaging |
| AdaBoostClassifier | Classification | Weak learners | Sequential re-weighting of errors |
| AdaBoostRegressor | Regression | Weak learners | Sequential re-weighting of errors |

---

## 🧠 Key Concepts

**Bagging (Bootstrap Aggregating)**
- Trains multiple independent models on random subsets of the training data
- Combines predictions by majority vote (classification) or averaging (regression)
- Best suited for high-variance models like decision trees
- Reduces overfitting without increasing bias

**AdaBoost (Adaptive Boosting)**
- Trains models sequentially, where each model focuses on samples the previous one got wrong
- Assigns higher weights to misclassified samples
- Combines weak learners into a strong overall predictor
- Effective at reducing bias

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations and null checks |
| **Scikit-learn** | Ensemble models, train-test split |
| **Seaborn** | Correlation heatmap visualization |
| **Matplotlib** | Plot rendering |
| **Google Colab** | Original development environment |

---

## 🔮 Possible Enhancements

- Add **Random Forest** and **Gradient Boosting** for a broader ensemble comparison
- Include **XGBoost** and **LightGBM** for state-of-the-art boosting performance
- Add **cross-validation** (k-fold) for more robust evaluation
- Compare models using **MAE, RMSE, R²** in addition to default score
- Visualize **feature importance** from ensemble models
- Add **hyperparameter tuning** with GridSearchCV or RandomizedSearchCV
- Convert back to a **Jupyter Notebook** for interactive analysis

---

## 📄 License

This project is open-source and available for educational and research use.

---
