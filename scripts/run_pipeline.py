#!/usr/bin/env python3
import os
import time
import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# local modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


def main(args):
    # MLflow target (project root /mlruns)
    project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
    mlruns_path = args.mlfow_uri or f"file://{project_root}/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    # Load & preprocess
    df = load_data(args.input)
    df = preprocess_data(df)

    # Split target
    target = args.target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data")

    # One-hot encode categorical features (excluding target)
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target]
    df_enc = build_features(df, categorical_cols=cat_cols)

    # Convert bool to int
    for c in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[c] = df_enc[c].astype(int)

    X = df_enc.drop(columns=[target])
    y = df_enc[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)

        # Predict
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        # Metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # Save model
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Console output
        print(f"\n⏱ train: {train_time:.2f}s | predict: {pred_time:.4f}s")
        print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlfow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")
    args = p.parse_args()
    main(args)
