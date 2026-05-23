"""
Step 3: Train XGBoost Credit Risk Model
- Loads credit_features.csv
- Trains XGBClassifier with scale_pos_weight
- Reports AUC-ROC and Gini coefficient
- Saves model to models/xgb_model.json
- Saves feature importance plot
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier, plot_importance

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH      = os.path.join(DATA_DIR,  "credit_features.csv")
MODEL_PATH     = os.path.join(MODEL_DIR, "xgb_model.json")
ENCODER_PATH   = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEAT_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")


# ── 1. Load & prepare ────────────────────────────────────────────────────────
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {df.shape}")

    # Drop non-model columns
    drop_cols = ["income_band", "tenure_band"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Label-encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save encoders
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # Save feature columns for dashboard use
    with open(FEAT_COLS_PATH, "w") as f:
        json.dump(list(X.columns), f)

    return X, y


# ── 2. Train ─────────────────────────────────────────────────────────────────
def train(X, y):
    counts = y.value_counts()
    imbalance_ratio = counts[0] / counts[1]
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}  →  scale_pos_weight={imbalance_ratio:.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    return model, X_train, X_test, y_train, y_test


# ── 3. Evaluate ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    print("\n── Model Performance ───────────────────────────────")
    print(f"  AUC-ROC : {auc:.4f}   (target ≥ 0.81)")
    print(f"  Gini    : {gini:.4f}   (= 2×AUC − 1)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Repaid','Default'])}")

    # Plots: ROC curve + Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Default"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix")

    plt.tight_layout()
    out = os.path.join(DATA_DIR, "model_evaluation.png")
    plt.savefig(out, dpi=120)
    print(f"\n  Evaluation plot saved → {out}")
    plt.close()

    return auc, gini


# ── 4. Feature importance ────────────────────────────────────────────────────
def save_feature_importance(model, X):
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.sort_values().plot(
        kind="barh", ax=ax, color="#3498db", edgecolor="white"
    )
    ax.set_title("Top 15 Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    out = os.path.join(DATA_DIR, "feature_importance.png")
    plt.savefig(out, dpi=120)
    print(f"  Feature importance plot saved → {out}")
    plt.close()

    print("\n── Top 10 Features ─────────────────────────────────")
    print(importance.head(10).round(4).to_string())


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y = load_and_prepare()
    model, X_train, X_test, y_train, y_test = train(X, y)

    auc, gini = evaluate(model, X_test, y_test)
    save_feature_importance(model, X)

    model.save_model(MODEL_PATH)
    print(f"\n✅  Model saved → {MODEL_PATH}")
    print(f"   AUC={auc:.4f}  |  Gini={gini:.4f}")
