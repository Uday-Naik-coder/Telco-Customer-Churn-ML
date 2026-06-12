import os
import pandas as pd
import mlflow

# === Which trained run to use (you already copied this under serving/model/<RUN_ID>) ===
RUN_ID = "3b1a41221fc44548aed629fa42b762e0"

# Paths to the copied MLflow artifacts for that run
ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "model",
    RUN_ID,
    "artifacts",
)
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")

# Load the logged model via MLflow (works regardless of underlying flavor)
model = mlflow.pyfunc.load_model(MODEL_DIR)

# Training-time feature order
with open(os.path.join(ARTIFACTS_DIR, "feature_columns.txt")) as f:
    FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

# Deterministic mappings for binary features (avoid single-row inference pitfalls)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror training-time transforms in a way that works for single-row inputs."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Coerce numerics
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply fixed binary mappings
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")        # safe nullable int
                .fillna(0)
                .astype(int)
            )

    # Any remaining object columns -> one-hot (drop_first=True like training)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Booleans to int if any appear
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Reorder to the exact training feature schema (missing -> 0)
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def predict(input_dict: dict):
    """
    Accepts raw customer dict, applies serving transforms, and returns the model prediction.
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    preds = model.predict(df_enc)
    # mlflow.pyfunc returns numpy array; make it plain
    if hasattr(preds, "tolist"):
        preds = preds.tolist()
    # return scalar if it’s a single value
    result = preds[0] if isinstance(preds, (list, tuple)) and len(preds) == 1 else preds

    # add churn message
    if result == 0:
        return "Not likely to churn"
    else:
        return "Likely to churn"




