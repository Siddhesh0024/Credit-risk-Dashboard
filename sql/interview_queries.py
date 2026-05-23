"""
Interview SQL Cheat Sheet — Credit Risk Project
Practice these until you can write them on a whiteboard.
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 1: Default rate by income band  (GROUP BY version — write this first)
# ─────────────────────────────────────────────────────────────────────────────
Q1_GROUPBY = """
SELECT
    income_band,
    COUNT(*)                    AS total,
    SUM(TARGET)                 AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM credit_data
WHERE income_band != 'nan'
GROUP BY income_band
ORDER BY default_rate_pct DESC;
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 2: Same result using WINDOW FUNCTIONS  (offer this unprompted — signals seniority)
# ─────────────────────────────────────────────────────────────────────────────
Q2_WINDOW = """
SELECT DISTINCT
    income_band,
    SUM(TARGET)  OVER (PARTITION BY income_band) AS band_defaults,
    COUNT(*)     OVER (PARTITION BY income_band) AS band_total,
    ROUND(
        100.0 * SUM(TARGET) OVER (PARTITION BY income_band)
              / COUNT(*)    OVER (PARTITION BY income_band),
        2
    ) AS default_rate_pct,
    ROUND(
        100.0 * COUNT(*) OVER (PARTITION BY income_band)
              / COUNT(*) OVER (),
        2
    ) AS pct_of_portfolio
FROM credit_data
WHERE income_band != 'nan'
ORDER BY default_rate_pct DESC;
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 3: Default rate by employment tenure  (very common live question)
# ─────────────────────────────────────────────────────────────────────────────
Q3_TENURE = """
SELECT
    CASE
        WHEN employment_years < 2  THEN '0-2 yrs'
        WHEN employment_years < 5  THEN '2-5 yrs'
        ELSE '5+ yrs'
    END AS tenure_band,
    COUNT(*)                    AS total,
    SUM(TARGET)                 AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM credit_data
GROUP BY tenure_band
ORDER BY default_rate_pct DESC;
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 4: Running cumulative default rate  (advanced — impressive if offered)
# ─────────────────────────────────────────────────────────────────────────────
Q4_RUNNING = """
SELECT
    income_band,
    defaults,
    total,
    default_rate_pct,
    SUM(defaults) OVER (ORDER BY default_rate_pct DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                        AS cumulative_defaults,
    SUM(total)    OVER (ORDER BY default_rate_pct DESC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                        AS cumulative_total
FROM (
    SELECT
        income_band,
        SUM(TARGET)                 AS defaults,
        COUNT(*)                    AS total,
        ROUND(AVG(TARGET)*100, 2)   AS default_rate_pct
    FROM credit_data
    WHERE income_band != 'nan'
    GROUP BY income_band
)
ORDER BY default_rate_pct DESC;
"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY 5: High-risk segment (delinquent + high DTI)
# ─────────────────────────────────────────────────────────────────────────────
Q5_HIGHRISK = """
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


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL QUERIES (for practice / review)
# ─────────────────────────────────────────────────────────────────────────────
import sqlite3, os, pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "credit.db")

QUERIES = {
    "1 · Default rate by income band (GROUP BY)": Q1_GROUPBY,
    "2 · Default rate by income band (WINDOW FN)": Q2_WINDOW,
    "3 · Default rate by tenure": Q3_TENURE,
    "4 · Running cumulative (advanced)": Q4_RUNNING,
    "5 · High-risk segments": Q5_HIGHRISK,
}

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("Run 4_sql_analysis.py first to create the database.")
    else:
        conn = sqlite3.connect(DB_PATH)
        for name, sql in QUERIES.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print('='*60)
            try:
                df = pd.read_sql_query(sql, conn)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        conn.close()
