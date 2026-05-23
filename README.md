# 🏦 Credit Risk Analytics Dashboard

A end-to-end credit risk analytics project using the **Home Credit Default Risk** dataset.  
Covers data engineering, XGBoost modelling, SQL analysis, and a live Streamlit dashboard.

---

## 📁 Project Structure

```
credit_risk_dashboard/
│
├── 1_download_data.py        ← Step 1: Download dataset from Kaggle
├── 2_eda_features.py         ← Step 2: EDA + feature engineering
├── 3_train_model.py          ← Step 3: Train XGBoost model
├── 4_sql_analysis.py         ← Step 4: Load SQLite + run SQL queries
├── dashboard.py              ← Step 5: Streamlit dashboard
│
├── sql/
│   └── interview_queries.py  ← All SQL queries (GROUP BY + window fn versions)
│
├── data/                     ← Created automatically
│   ├── application_train.csv
│   ├── credit_features.csv
│   ├── credit.db
│   ├── eda_plots.png
│   ├── model_evaluation.png
│   └── feature_importance.png
│
├── models/                   ← Created automatically
│   ├── xgb_model.json
│   ├── label_encoders.pkl
│   └── feature_columns.json
│
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.8 or higher → [python.org](https://www.python.org/downloads/)
- VS Code → [code.visualstudio.com](https://code.visualstudio.com/)
- VS Code extensions: **Python** (Microsoft) + **Jupyter**

### 2. Clone / create the project folder

```bash
# Open VS Code terminal (Ctrl+`)
cd your-projects-folder
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project (Step by Step)

### Step 1 — Download the Data

**Option A: Kaggle API (automated)**

1. Go to [kaggle.com](https://www.kaggle.com/) → Account → **Create New API Token**
2. This downloads `kaggle.json` — place it at `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\YourName\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   python 1_download_data.py
   ```

**Option B: Manual download**

1. Go to [kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk)
2. Download `application_train.csv`
3. Place it inside the `data/` folder (create the folder if it doesn't exist)

---

### Step 2 — EDA & Feature Engineering

```bash
python 2_eda_features.py
```

**What it does:**
- Loads `application_train.csv` (~300K rows)
- Prints class distribution and imbalance ratio
- Engineers 10 new features (debt-to-income, credit utilisation, etc.)
- Saves cleaned data to `data/credit_features.csv`
- Saves EDA plot to `data/eda_plots.png`

**Key engineered features:**

| Feature | Formula | Business meaning |
|---|---|---|
| `debt_to_income` | AMT_ANNUITY / AMT_INCOME_TOTAL | Repayment burden relative to income |
| `credit_utilisation` | AMT_CREDIT / AMT_GOODS_PRICE | How much credit vs. goods value |
| `income_per_person` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Household-adjusted income |
| `employment_years` | DAYS_EMPLOYED / -365 | Tenure at current employer |
| `delinquency_flag` | DAYS_CREDIT_DPD > 0 | Binary: any prior late payment |

---

### Step 3 — Train the XGBoost Model

```bash
python 3_train_model.py
```

**What it does:**
- Encodes categorical columns with LabelEncoder
- Trains XGBClassifier with `scale_pos_weight` to handle ~11:1 class imbalance
- Prints AUC-ROC and Gini coefficient
- Saves model to `models/xgb_model.json`
- Saves feature importance plot to `data/feature_importance.png`

**Expected output:**
```
AUC-ROC : 0.81+   (target ≥ 0.81)
Gini    : 0.62+   (= 2×AUC − 1)
```

**Why XGBoost over Logistic Regression?**
- Handles non-linear relationships in financial data
- Built-in `scale_pos_weight` for imbalanced classes
- More robust to outliers (common in income/credit data)
- Feature importance is interpretable for business stakeholders

---

### Step 4 — SQL Analysis Layer

```bash
python 4_sql_analysis.py
```

**What it does:**
- Loads `credit_features.csv` into a local SQLite database (`data/credit.db`)
- Runs 6 analysis queries and prints results
- Saves all results to `data/sql_results.csv`

**No extra software needed** — SQLite is built into Python.

**Key queries run:**

```sql
-- Default rate by income band
SELECT income_band,
       COUNT(*)                    AS total,
       SUM(TARGET)                 AS defaults,
       ROUND(AVG(TARGET)*100, 2)   AS default_rate_pct
FROM credit_data
WHERE income_band != 'nan'
GROUP BY income_band
ORDER BY default_rate_pct DESC;
```

```sql
-- Window function version (asked in interviews)
SELECT DISTINCT income_band,
       SUM(TARGET) OVER (PARTITION BY income_band) AS band_defaults,
       ROUND(100.0 * SUM(TARGET) OVER (PARTITION BY income_band)
             / COUNT(*) OVER (PARTITION BY income_band), 2) AS default_rate_pct
FROM credit_data
WHERE income_band != 'nan';
```

---

### Step 5 — Launch the Dashboard

```bash
streamlit run dashboard.py
```

Opens at **http://localhost:8501** in your browser.

**Dashboard features:**
- 📊 KPI cards: total applicants, defaults, default rate, avg DTI
- 📉 Default rate by income band (bar chart)
- 📉 Default rate by employment tenure (horizontal bar)
- 🔑 Top 12 feature importances (XGBoost)
- 🗄️ Interactive SQL explorer with 4 preset queries
- 🔮 Live applicant scorer with gauge chart and risk badge

---

## 🗄️ SQL Practice (Interview Prep)

Run all interview SQL queries at once:

```bash
python sql/interview_queries.py
```

This runs 5 queries including the window function version.  
Practice writing `Q3_TENURE` from scratch — it's the most common live question.

---

## 📊 Model Performance

| Metric | Value | Notes |
|---|---|---|
| AUC-ROC | ≥ 0.81 | Industry benchmark for credit scoring |
| Gini | ≥ 0.62 | = 2 × AUC − 1 |
| scale_pos_weight | ~11 | Handles 11:1 class imbalance |

---

## 🎯 Interview Q&A

**Q: What is AUC-ROC and why use it over accuracy?**  
A: AUC measures the model's ability to rank defaulters above non-defaulters across all thresholds — threshold-agnostic. Accuracy is misleading on imbalanced data (a model predicting "no default" for everyone gets 92% accuracy but is useless).

**Q: What is the Gini coefficient here?**  
A: Gini = 2 × AUC − 1. It rescales AUC to a 0–1 range. An AUC of 0.81 → Gini of 0.62. Industry minimum for production credit models is typically 0.40.

**Q: Why XGBoost over Logistic Regression?**  
A: XGBoost handles non-linearity, missing values, and imbalanced classes natively. Logistic Regression assumes linear relationships and needs extensive manual feature engineering. In practice, XGBoost gives 5–10% higher AUC on tabular credit data.

**Q: Which 3 features mattered most and why does that make business sense?**  
A: Typically `debt_to_income` (repayment burden), `employment_years` (income stability), and `age_years` (credit history length). All three directly predict the capacity and willingness to repay.

**Q: A bank uses your model and rejects a good customer — what went wrong?**  
A: A false positive (type I error). This means the model's decision threshold is too conservative. The fix is to tune the threshold based on the business cost matrix: what's the cost of rejecting a good customer (lost revenue) vs. approving a bad one (write-off)? We don't just maximise accuracy — we minimise expected loss.

---

## 📦 Dependencies

```
pandas · numpy · xgboost · scikit-learn
matplotlib · seaborn · streamlit · shap · kaggle
```

Install all: `pip install -r requirements.txt`

---

## 📝 Notes

- The `delinquency_flag` feature is set to 0 if `DAYS_CREDIT_DPD` is absent from `application_train.csv` — this column lives in the bureau file. To enrich the model, join with `bureau.csv` from the same Kaggle competition.
- SQLite is used for portability. The same queries run unchanged in PostgreSQL or BigQuery — just change the connection string.
- SHAP explainability can be added with `import shap; shap.TreeExplainer(model)` for per-applicant explanation.
