import os
import pandas as pd
import mlflow

# Model folder packaged inside the container
MODEL_DIR = "/app/model"

# Load the logged model via MLflow
model = mlflow.pyfunc.load_model(MODEL_DIR)

# Load training-time feature order
with open(os.path.join(MODEL_DIR, "feature_columns.txt")) as f:
    FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

# Deterministic mappings for binary features
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
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # One-hot encode categorical object columns
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Convert booleans to ints
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align with training feature schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def predict(input_dict: dict):
    """Accepts raw customer dict, applies serving transforms, and returns model prediction."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    preds = model.predict(df_enc)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    result = preds[0] if isinstance(preds, (list, tuple)) and len(preds) == 1 else preds

    return "Likely to churn" if result == 1 else "Not likely to churn"
