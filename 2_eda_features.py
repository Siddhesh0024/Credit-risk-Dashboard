"""
Step 2: EDA & Feature Engineering
- Loads application_train.csv
- Prints class imbalance stats
- Engineers key features
- Saves cleaned data to data/credit_features.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_PATH = os.path.join(DATA_DIR, "application_train.csv")
OUT_PATH = os.path.join(DATA_DIR, "credit_features.csv")


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Shape: {df.shape}")
    return df


# ── 2. EDA ───────────────────────────────────────────────────────────────────
def run_eda(df):
    print("\n── Class Distribution ──────────────────────────────")
    counts = df["TARGET"].value_counts()
    print(counts)
    imbalance_ratio = counts[0] / counts[1]
    print(f"  Imbalance ratio (neg/pos): {imbalance_ratio:.1f}x")

    print("\n── Missing values (top 10 cols) ────────────────────")
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(missing.head(10).round(1).to_string())

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[0].set_title("Class Distribution (TARGET)")
    axes[0].set_xticklabels(["Repaid (0)", "Default (1)"], rotation=0)
    axes[0].set_ylabel("Count")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=10)

    # Income distribution by target
    df[df["TARGET"] == 0]["AMT_INCOME_TOTAL"].clip(upper=500000).plot(
        kind="hist", bins=50, ax=axes[1], alpha=0.6, label="Repaid", color="#2ecc71")
    df[df["TARGET"] == 1]["AMT_INCOME_TOTAL"].clip(upper=500000).plot(
        kind="hist", bins=50, ax=axes[1], alpha=0.6, label="Default", color="#e74c3c")
    axes[1].set_title("Income Distribution by Default Status")
    axes[1].set_xlabel("Annual Income (clipped at 500K)")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(DATA_DIR, "eda_plots.png")
    plt.savefig(out, dpi=120)
    print(f"\n  EDA plot saved → {out}")
    plt.close()

    return imbalance_ratio


# ── 3. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df):
    print("\n── Engineering features ────────────────────────────")

    # Core ratio features
    df["debt_to_income"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["credit_utilisation"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["income_per_person"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["annuity_to_credit"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)

    # Age & employment (DAYS columns are negative integers)
    df["age_years"] = df["DAYS_BIRTH"] / -365
    df["employment_years"] = df["DAYS_EMPLOYED"].apply(
        lambda x: x / -365 if x < 0 else 0)  # positive DAYS_EMPLOYED = pensioner/unemployed

    # Delinquency flag (from bureau data proxy)
    if "DAYS_CREDIT_DPD" in df.columns:
        df["delinquency_flag"] = (df["DAYS_CREDIT_DPD"] > 0).astype(int)
    else:
        df["delinquency_flag"] = 0  # column absent in application_train; set to 0

    # Document submission score (how many documents provided)
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["doc_submission_score"] = df[doc_cols].sum(axis=1)

    # Income band (for SQL layer)
    df["income_band"] = pd.cut(
        df["AMT_INCOME_TOTAL"],
        bins=[0, 50000, 100000, 200000, 500000, float("inf")],
        labels=["<50K", "50-100K", "100-200K", "200-500K", "500K+"]
    ).astype(str)

    # Employment tenure band (for SQL layer)
    df["tenure_band"] = pd.cut(
        df["employment_years"],
        bins=[-0.01, 2, 5, 100],
        labels=["0-2 yrs", "2-5 yrs", "5+ yrs"]
    ).astype(str)

    engineered = [
        "debt_to_income", "credit_utilisation", "income_per_person",
        "annuity_to_credit", "age_years", "employment_years",
        "delinquency_flag", "doc_submission_score", "income_band", "tenure_band"
    ]
    print(f"  Created features: {engineered}")
    return df


# ── 4. Select & clean final columns ─────────────────────────────────────────
FEATURE_COLS = [
    # Original
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    # Engineered
    "debt_to_income", "credit_utilisation", "income_per_person",
    "annuity_to_credit", "age_years", "employment_years",
    "delinquency_flag", "doc_submission_score",
    "income_band", "tenure_band",
    # Target
    "TARGET",
]

def select_and_clean(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols].copy()

    # Median imputation for numerics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "TARGET"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Mode imputation for categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    print(f"\n  Final dataset shape: {df.shape}")
    print(f"  Null values remaining: {df.isnull().sum().sum()}")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    imbalance_ratio = run_eda(df)
    df = engineer_features(df)
    df = select_and_clean(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅  Clean feature file saved → {OUT_PATH}")
    print(f"   Imbalance ratio to use in XGBoost: {imbalance_ratio:.1f}")
