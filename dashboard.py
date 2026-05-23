"""
Step 5: Streamlit Credit Risk Analytics Dashboard
Run with:  streamlit run dashboard.py
"""

import os
import json
import pickle
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from xgboost import XGBClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE, "data")
MODEL_DIR  = os.path.join(BASE, "models")
DB_PATH    = os.path.join(DATA_DIR,  "credit.db")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
ENC_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEAT_PATH  = os.path.join(MODEL_DIR, "feature_columns.json")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load assets (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_encoders():
    with open(ENC_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_feature_cols():
    with open(FEAT_PATH) as f:
        return json.load(f)

@st.cache_data
def load_db_data(income_filter, loan_filter):
    conn = sqlite3.connect(DB_PATH)
    where = []
    if income_filter != "All":
        where.append(f"income_band = '{income_filter}'")
    if loan_filter != "All":
        where.append(f"NAME_CONTRACT_TYPE = '{loan_filter}'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    df = pd.read_sql_query(f"SELECT * FROM credit_data {where_sql} LIMIT 10000", conn)
    conn.close()
    return df

@st.cache_data
def get_filter_options():
    conn = sqlite3.connect(DB_PATH)
    income_bands = pd.read_sql_query(
        "SELECT DISTINCT income_band FROM credit_data WHERE income_band != 'nan' ORDER BY income_band",
        conn)["income_band"].tolist()
    loan_types = pd.read_sql_query(
        "SELECT DISTINCT NAME_CONTRACT_TYPE FROM credit_data ORDER BY NAME_CONTRACT_TYPE",
        conn)["NAME_CONTRACT_TYPE"].tolist()
    conn.close()
    return ["All"] + income_bands, ["All"] + loan_types
@st.cache_data
def load_data():
    if os.path.exists("data/credit.db"):
        import sqlite3
        conn = sqlite3.connect("data/credit.db")
        df = pd.read_sql_query("SELECT * FROM credit_data", conn)
        conn.close()
    else:
        df = pd.read_csv("data/sample_data.csv")
    return df

# ── Helpers ──────────────────────────────────────────────────────────────────
def risk_badge(prob):
    if prob >= 0.6:
        return "🔴 HIGH RISK"
    elif prob >= 0.35:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def score_color(prob):
    if prob >= 0.6:   return "#e74c3c"
    elif prob >= 0.35: return "#f39c12"
    else:             return "#27ae60"

def gauge_html(prob):
    pct   = int(prob * 100)
    color = score_color(prob)
    return f"""
    <div style="text-align:center; padding: 12px 0;">
      <svg width="200" height="110" viewBox="0 0 200 110">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e0e0e0" stroke-width="18" stroke-linecap="round"/>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="18"
              stroke-linecap="round" stroke-dasharray="{pct * 2.51:.1f} 251"
              style="transition: stroke-dasharray 0.5s ease;"/>
        <text x="100" y="95" text-anchor="middle" font-size="28" font-weight="bold" fill="{color}">{pct}%</text>
        <text x="100" y="112" text-anchor="middle" font-size="11" fill="#888">Default Probability</text>
      </svg>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏦 Credit Risk Analytics")
st.sidebar.markdown("---")

income_options, loan_options = get_filter_options()

st.sidebar.subheader("📊 Dashboard Filters")
income_filter = st.sidebar.selectbox("Income Band", income_options)
loan_filter   = st.sidebar.selectbox("Loan Type",   loan_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Score an Applicant")

with st.sidebar.expander("Enter applicant details", expanded=True):
    amt_income    = st.number_input("Annual Income (₹)",     10000, 5000000, 200000, step=10000)
    amt_credit    = st.number_input("Loan Amount (₹)",       10000, 5000000, 500000, step=10000)
    amt_annuity   = st.number_input("Annual Repayment (₹)",  5000,  500000,  30000,  step=1000)
    amt_goods     = st.number_input("Goods Price (₹)",       5000,  5000000, 450000, step=10000)
    age_years     = st.slider("Age (years)",                 18, 70, 35)
    emp_years     = st.slider("Employment Years",            0, 40, 5)
    cnt_fam       = st.slider("Family Members",              1, 10, 3)
    delinquent    = st.checkbox("Prior delinquency?")
    contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    gender        = st.selectbox("Gender", ["M", "F"])
    education     = st.selectbox("Education", [
        "Secondary / secondary special", "Higher education",
        "Incomplete higher", "Lower secondary", "Academic degree"])

score_btn = st.sidebar.button("🎯 Score This Applicant", use_container_width=True)


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("Credit Risk Analytics Dashboard")
st.caption("Home Credit Default Risk · XGBoost · AUC-ROC ≥ 0.81")

# ── KPI row ──────────────────────────────────────────────────────────────────
df_view = load_db_data(income_filter, loan_filter)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Applicants",    f"{len(df_view):,}")
with k2:
    n_default = int(df_view["TARGET"].sum())
    st.metric("Defaults in View",    f"{n_default:,}")
with k3:
    rate = df_view["TARGET"].mean() * 100
    st.metric("Default Rate",        f"{rate:.1f}%")
with k4:
    avg_dti = df_view["debt_to_income"].mean() if "debt_to_income" in df_view.columns else 0
    st.metric("Avg Debt-to-Income",  f"{avg_dti:.3f}")

st.markdown("---")

# ── Charts row ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Default Rate by Income Band")
    conn = sqlite3.connect(DB_PATH)
    band_sql = """
        SELECT income_band,
               ROUND(AVG(TARGET)*100,2) AS default_rate_pct,
               COUNT(*) AS total
        FROM credit_data
        WHERE income_band != 'nan'
        GROUP BY income_band
        ORDER BY default_rate_pct DESC
    """
    band_df = pd.read_sql_query(band_sql, conn)
    conn.close()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(band_df["income_band"], band_df["default_rate_pct"],
                  color=["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db"][:len(band_df)],
                  edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Income Band")
    ax.set_title("Default Rate by Income Band")
    for bar, val in zip(bars, band_df["default_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Default Rate by Employment Tenure")
    conn = sqlite3.connect(DB_PATH)
    tenure_df = pd.read_sql_query("""
        SELECT tenure_band,
               ROUND(AVG(TARGET)*100,2) AS default_rate_pct,
               COUNT(*) AS total
        FROM credit_data
        WHERE tenure_band != 'nan'
        GROUP BY tenure_band
        ORDER BY default_rate_pct DESC
    """, conn)
    conn.close()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#e74c3c","#f39c12","#27ae60"][:len(tenure_df)]
    ax.barh(tenure_df["tenure_band"], tenure_df["default_rate_pct"],
            color=colors, edgecolor="white")
    ax.set_xlabel("Default Rate (%)")
    ax.set_title("Default Rate by Employment Tenure")
    for i, (val, total) in enumerate(zip(tenure_df["default_rate_pct"], tenure_df["total"])):
        ax.text(val + 0.05, i, f"{val}%  (n={total:,})", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔑 Top Feature Importances")

try:
    model = load_model()
    feat_cols = load_feature_cols()
    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(10, 4))
    importance.sort_values().plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Top 12 Features (XGBoost Importance)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
except Exception as e:
    st.info(f"Train the model first (run 3_train_model.py). {e}")

# ── SQL explorer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗄️ SQL Query Explorer")

PRESET_QUERIES = {
    "Default rate by income band": """
SELECT income_band,
       COUNT(*)                    AS total,
       SUM(TARGET)                 AS defaults,
       ROUND(AVG(TARGET)*100, 2)   AS default_rate_pct
FROM credit_data
WHERE income_band != 'nan'
GROUP BY income_band
ORDER BY default_rate_pct DESC;""",

    "Default rate by tenure (window fn)": """
SELECT DISTINCT tenure_band,
       SUM(TARGET) OVER (PARTITION BY tenure_band)  AS band_defaults,
       COUNT(*)    OVER (PARTITION BY tenure_band)  AS band_total,
       ROUND(100.0 * SUM(TARGET) OVER (PARTITION BY tenure_band)
             / COUNT(*) OVER (PARTITION BY tenure_band), 2) AS default_rate_pct
FROM credit_data
WHERE tenure_band != 'nan'
ORDER BY default_rate_pct DESC;""",

    "High-risk segments": """
SELECT income_band, tenure_band,
       COUNT(*)                        AS applicants,
       SUM(TARGET)                     AS defaults,
       ROUND(AVG(TARGET)*100, 2)       AS default_rate_pct,
       ROUND(AVG(debt_to_income), 3)   AS avg_debt_to_income
FROM credit_data
WHERE delinquency_flag = 1 AND debt_to_income > 0.3
GROUP BY income_band, tenure_band
HAVING COUNT(*) >= 20
ORDER BY default_rate_pct DESC
LIMIT 10;""",

    "Default by loan type": """
SELECT NAME_CONTRACT_TYPE             AS loan_type,
       COUNT(*)                       AS total,
       SUM(TARGET)                    AS defaults,
       ROUND(AVG(TARGET)*100, 2)      AS default_rate_pct
FROM credit_data
GROUP BY loan_type;""",
}

chosen = st.selectbox("Preset queries", list(PRESET_QUERIES.keys()))
query  = st.text_area("SQL Query", value=PRESET_QUERIES[chosen], height=140)

if st.button("▶ Run Query"):
    try:
        conn = sqlite3.connect(DB_PATH)
        result_df = pd.read_sql_query(query, conn)
        conn.close()
        st.dataframe(result_df, use_container_width=True)
        st.caption(f"{len(result_df)} rows returned")
    except Exception as e:
        st.error(f"SQL error: {e}")

# ── Applicant Scorer ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔮 Applicant Risk Scorer")

if score_btn:
    try:
        model    = load_model()
        encoders = load_encoders()
        feat_cols = load_feature_cols()

        # Build feature row matching training columns
        row = {
            "AMT_INCOME_TOTAL"   : amt_income,
            "AMT_CREDIT"         : amt_credit,
            "AMT_ANNUITY"        : amt_annuity,
            "AMT_GOODS_PRICE"    : amt_goods,
            "DAYS_BIRTH"         : int(-age_years * 365),
            "DAYS_EMPLOYED"      : int(-emp_years * 365),
            "CNT_FAM_MEMBERS"    : cnt_fam,
            "NAME_CONTRACT_TYPE" : contract_type,
            "CODE_GENDER"        : gender,
            "NAME_INCOME_TYPE"   : "Working",
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS" : "Married",
            "debt_to_income"     : amt_annuity / (amt_income + 1),
            "credit_utilisation" : amt_credit / (amt_goods + 1),
            "income_per_person"  : amt_income / (cnt_fam + 1),
            "annuity_to_credit"  : amt_annuity / (amt_credit + 1),
            "age_years"          : age_years,
            "employment_years"   : emp_years,
            "delinquency_flag"   : int(delinquent),
            "doc_submission_score": 5,
        }

        df_row = pd.DataFrame([row])

        # Encode categoricals
        for col, le in encoders.items():
            if col in df_row.columns:
                val = df_row[col].astype(str).iloc[0]
                if val in le.classes_:
                    df_row[col] = le.transform([val])[0]
                else:
                    df_row[col] = le.transform([le.classes_[0]])[0]

        # Align with training columns
        for c in feat_cols:
            if c not in df_row.columns:
                df_row[c] = 0
        df_row = df_row[feat_cols]

        prob = float(model.predict_proba(df_row)[0, 1])

        scol1, scol2 = st.columns([1, 2])
        with scol1:
            st.markdown(gauge_html(prob), unsafe_allow_html=True)
            badge = risk_badge(prob)
            color = score_color(prob)
            st.markdown(
                f"<div style='text-align:center; font-size:20px; font-weight:bold; color:{color}'>{badge}</div>",
                unsafe_allow_html=True
            )

        with scol2:
            st.markdown("**Key ratios for this applicant:**")
            ratios = {
                "Debt-to-Income"      : f"{amt_annuity/(amt_income+1):.3f}  {'⚠️' if amt_annuity/amt_income > 0.4 else '✅'}",
                "Credit Utilisation"  : f"{amt_credit/(amt_goods+1):.3f}",
                "Income per person"   : f"₹{amt_income/(cnt_fam+1):,.0f}",
                "Age"                 : f"{age_years} yrs",
                "Employment tenure"   : f"{emp_years} yrs  {'⚠️ Short' if emp_years < 2 else '✅'}",
                "Prior delinquency"   : "⚠️ Yes" if delinquent else "✅ No",
            }
            for k, v in ratios.items():
                st.write(f"**{k}:** {v}")

    except FileNotFoundError:
        st.warning("Model not found. Run `python 3_train_model.py` first.")
    except Exception as e:
        st.error(f"Scoring error: {e}")
else:
    st.info("👈 Fill in applicant details in the sidebar and click **Score This Applicant**.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with XGBoost · SQLite · Streamlit  |  Home Credit Default Risk Dataset")
