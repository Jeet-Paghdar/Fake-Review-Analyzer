"""
train_model.py
Trains a TF-IDF + Logistic Regression pipeline on the synthetic review dataset.
Saves model artifacts so app.py can load them without retraining.
Run: python train_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib

from generate_data import generate_dataset

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score
)
from sklearn.pipeline import Pipeline


def train():
    print("=" * 55)
    print("  Fake Review Forensic Analyzer — Model Training")
    print("=" * 55)

    # ── 1. Load / generate data ────────────────────────────────
    if os.path.exists("reviews_dataset.csv"):
        print("\n📂 Loading existing dataset …")
        df = pd.read_csv("reviews_dataset.csv")
    else:
        print("\n⚙️  Generating synthetic dataset …")
        df = generate_dataset()
        df.to_csv("reviews_dataset.csv", index=False)

    print(f"   Total samples : {len(df)}")
    print(f"   Fake          : {df['label'].sum()}")
    print(f"   Authentic     : {(df['label'] == 0).sum()}")

    X = df["review"].astype(str).to_numpy()
    y = df["label"].astype(int).to_numpy()

    # ── 2. Train / test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Build Pipeline ──────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=15000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    # ── 4. Cross-validation ────────────────────────────────────
    print("\n🔄 Running 5-fold cross-validation …")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"   CV Accuracy : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

    # ── 5. Final fit ───────────────────────────────────────────
    print("\n🚀 Training final model …")
    pipeline.fit(X_train, y_train)

    # ── 6. Evaluation ──────────────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Test Accuracy : {acc:.4f}")
    print(f"   ROC-AUC      : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Authentic", "Fake"]))

    # ── 7. Save artifacts ─────────────────────────────────────
    joblib.dump(pipeline, "review_model.pkl")
    print("✅ Model saved → review_model.pkl")

    # Save class names for the dashboard
    meta = {"classes": ["Authentic", "Fake"], "accuracy": acc, "auc": auc}
    joblib.dump(meta, "model_meta.pkl")
    print("✅ Metadata saved → model_meta.pkl")
    print("\n🎉 Training complete! Run  →  streamlit run app.py")


if __name__ == "__main__":
    train()
# Finalized model training pipeline
