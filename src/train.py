import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

EVAL_THRESHOLD = 0.70


def _load_training_data(data_path: str) -> pd.DataFrame:
    """Load training data and optionally append phase 2 when using the default lab path."""
    df_train = pd.read_csv(data_path)

    if os.path.basename(data_path) == "train_phase1.csv" and len(df_train) <= 3000:
        phase2_path = os.path.join(os.path.dirname(data_path), "train_phase2.csv")
        if os.path.exists(phase2_path):
            df_phase2 = pd.read_csv(phase2_path)
            df_train = pd.concat([df_train, df_phase2], ignore_index=True)

    return df_train


def check_label_distribution(y_train: pd.Series) -> dict:
    """Check if any label in the training set has less than 10% representation."""
    dist = y_train.value_counts(normalize=True).to_dict()
    for label, prop in dist.items():
        if prop < 0.1:
            warnings.warn(f"Warning: Data skew detected! Class {label} only accounts for {prop:.2%} of training data.")
            print(f"[WARNING] Data skew: Class {label} accounts for {prop:.2%} (<10%).")
    return dist


def get_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    # Loai bo model_type ra khoi params de pass vao sklearn
    model_params = {k: v for k, v in params.items() if k != "model_type"}
    
    if model_type == "random_forest":
        return RandomForestClassifier(**{k: v for k, v in model_params.items() if k in ["n_estimators", "max_depth", "min_samples_split"]}, random_state=42)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**{k: v for k, v in model_params.items() if k in ["n_estimators", "max_depth", "min_samples_split"]}, random_state=42)
    elif model_type == "logistic_regression":
        return LogisticRegression(**{k: v for k, v in model_params.items() if k in ["max_iter"]}, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def generate_report(y_eval, preds, acc, f1, report_path="outputs/report.txt"):
    """Calculate confusion matrix, precision, recall and write to a text report."""
    cm = confusion_matrix(y_eval, preds)
    precision = precision_score(y_eval, preds, average=None)
    recall = recall_score(y_eval, preds, average=None)
    
    with open(report_path, "w") as f:
        f.write("=== AUTOMATED PERFORMANCE REPORT ===\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (Weighted): {f1:.4f}\n\n")
        f.write("--- Metrics per class ---\n")
        for i in range(len(precision)):
            f.write(f"Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}\n")
        
        f.write("\n--- Confusion Matrix ---\n")
        f.write(str(cm) + "\n")
    print(f"Report saved to {report_path}")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    df_train = _load_training_data(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    dist = check_label_distribution(y_train)

    with mlflow.start_run():
        mlflow.log_params(params)

        model = get_model(params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = float(accuracy_score(y_eval, preds))
        f1 = float(f1_score(y_eval, preds, average="weighted"))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        generate_report(y_eval, preds, acc, f1)

        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, 
                "f1_score": f1,
                "label_distribution": dist
            }, f, indent=4)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
