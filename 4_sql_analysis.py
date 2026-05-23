"""
Step 4: SQL Analysis Layer
- Loads credit_features.csv into a SQLite database
- Runs all key SQL queries (GROUP BY + WINDOW FUNCTION versions)
- Exports results to data/sql_results.csv
"""

import os
import sqlite3
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH  = os.path.join(DATA_DIR, "credit.db")
CSV_PATH = os.path.join(DATA_DIR, "credit_features.csv")


def build_database():
    print("Loading CSV into SQLite...")
    df = pd.read_csv(CSV_PATH)

    # Keep only columns useful for SQL analysis
    sql_cols = [
        "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "debt_to_income", "credit_utilisation", "age_years",
        "employment_years", "delinquency_flag", "income_band", "tenure_band",
    ]
    sql_cols = [c for c in sql_cols if c in df.columns]
    df = df[sql_cols]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("credit_data", conn, if_exists="replace", index=False)
    print(f"  {len(df):,} rows loaded into {DB_PATH}")
    return conn


def run_queries(conn):
    results = {}

    # ── Q1: Default rate by income band (GROUP BY) ───────────────────────────
    q1 = """
    SELECT
        income_band,
        COUNT(*)                        AS total,
        SUM(TARGET)                     AS defaults,
        ROUND(AVG(TARGET) * 100, 2)     AS default_rate_pct
    FROM credit_data
    WHERE income_band != 'nan'
    GROUP BY income_band
    ORDER BY default_rate_pct DESC;
    """

    # ── Q2: Default rate by employment tenure band ───────────────────────────
    q2 = """
    SELECT
        tenure_band,
        COUNT(*)                        AS total,
        SUM(TARGET)                     AS defaults,
        ROUND(AVG(TARGET) * 100, 2)     AS default_rate_pct
    FROM credit_data
    WHERE tenure_band != 'nan'
    GROUP BY tenure_band
    ORDER BY default_rate_pct DESC;
    """

    # ── Q3: Default rate by loan type ────────────────────────────────────────
    q3 = """
    SELECT
        NAME_CONTRACT_TYPE              AS loan_type,
        COUNT(*)                        AS total,
        SUM(TARGET)                     AS defaults,
        ROUND(AVG(TARGET) * 100, 2)     AS default_rate_pct
    FROM credit_data
    GROUP BY loan_type
    ORDER BY default_rate_pct DESC;
    """

    # ── Q4: Default rate by education level ──────────────────────────────────
    q4 = """
    SELECT
        NAME_EDUCATION_TYPE             AS education,
        COUNT(*)                        AS total,
        SUM(TARGET)                     AS defaults,
        ROUND(AVG(TARGET) * 100, 2)     AS default_rate_pct
    FROM credit_data
    GROUP BY education
    ORDER BY default_rate_pct DESC;
    """

    # ── Q5: Window function version — default rate by income band ────────────
    # (This is what interviewers ask for on the spot)
    q5 = """
    SELECT DISTINCT
        income_band,
        SUM(TARGET) OVER (PARTITION BY income_band)   AS band_defaults,
        COUNT(*)    OVER (PARTITION BY income_band)   AS band_total,
        ROUND(
            100.0 * SUM(TARGET) OVER (PARTITION BY income_band) /
                    COUNT(*)    OVER (PARTITION BY income_band),
            2
        )                                             AS default_rate_pct,
        ROUND(
            100.0 * COUNT(*) OVER (PARTITION BY income_band) /
                    COUNT(*) OVER (),
            2
        )                                             AS pct_of_portfolio
    FROM credit_data
    WHERE income_band != 'nan'
    ORDER BY default_rate_pct DESC;
    """

    # ── Q6: High-risk segment — delinquent + high debt-to-income ─────────────
    q6 = """
    SELECT
        income_band,
        tenure_band,
        COUNT(*)                        AS applicants,
        SUM(TARGET)                     AS defaults,
        ROUND(AVG(TARGET) * 100, 2)     AS default_rate_pct,
        ROUND(AVG(debt_to_income), 3)   AS avg_debt_to_income
    FROM credit_data
    WHERE delinquency_flag = 1
      AND debt_to_income > 0.3
    GROUP BY income_band, tenure_band
    HAVING COUNT(*) >= 50
    ORDER BY default_rate_pct DESC
    LIMIT 10;
    """

    queries = {
        "default_by_income_band"    : q1,
        "default_by_tenure"         : q2,
        "default_by_loan_type"      : q3,
        "default_by_education"      : q4,
        "window_fn_income_band"     : q5,
        "high_risk_segments"        : q6,
    }

    print("\n── SQL Query Results ───────────────────────────────")
    for name, sql in queries.items():
        df = pd.read_sql_query(sql, conn)
        results[name] = df
        print(f"\n[{name}]")
        print(df.to_string(index=False))

    # Save all results to one Excel-like CSV (each query appended with header)
    out = os.path.join(DATA_DIR, "sql_results.csv")
    with open(out, "w") as f:
        for name, df in results.items():
            f.write(f"\n### {name}\n")
            df.to_csv(f, index=False)
    print(f"\n✅  SQL results saved → {out}")

    return results


if __name__ == "__main__":
    conn = build_database()
    run_queries(conn)
    conn.close()
