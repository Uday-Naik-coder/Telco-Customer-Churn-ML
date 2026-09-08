"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================
"""

import os
import glob
from pathlib import Path

import pandas as pd
import mlflow


# ============================================================
# MODEL LOADING CONFIGURATION
# ============================================================

# Docker location
DOCKER_MODEL_DIR = Path("/app/model")

# Local project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model():
    """
    Load the MLflow model.

    Production:
        /app/model

    Local development:
        Latest model found under ./mlruns/*/*/artifacts/model
    """

    # --------------------------------------------------------
    # 1. Try Docker/production location
    # --------------------------------------------------------
    try:
        model = mlflow.pyfunc.load_model(str(DOCKER_MODEL_DIR))
        print(f"[OK] Model loaded successfully from {DOCKER_MODEL_DIR}")

        return model, DOCKER_MODEL_DIR

    except Exception as e:
        print(f"[ERROR] Failed to load model from {DOCKER_MODEL_DIR}: {e}")

    # --------------------------------------------------------
    # 2. Fallback to local MLflow artifacts
    # --------------------------------------------------------
    local_model_paths = glob.glob(
        str(PROJECT_ROOT / "mlruns" / "*" / "*" / "artifacts" / "model")
    )

    if not local_model_paths:
        raise Exception(
            "No MLflow model found in local mlruns directory."
        )

    # Pick the most recently modified model
    latest_model = max(
        local_model_paths,
        key=os.path.getmtime
    )

    try:
        model = mlflow.pyfunc.load_model(latest_model)

        model_dir = Path(latest_model)

        print(f"[OK] Fallback: Loaded model from {model_dir}")

        return model, model_dir

    except Exception as e:
        raise Exception(
            f"Failed to load local MLflow model: {e}"
        )


model, MODEL_DIR = load_model()


# ============================================================
# FEATURE SCHEMA LOADING
# ============================================================

try:
    # IMPORTANT:
    #
    # MLflow artifacts look like:
    #
    # artifacts/
    # ├── feature_columns.txt
    # ├── preprocessing.pkl
    # └── model/
    #     ├── model.pkl
    #     └── MLmodel
    #
    # MODEL_DIR points to:
    #
    # artifacts/model/
    #
    # Docker copies feature_columns.txt INTO the model dir,
    # but MLflow stores it one level above (in artifacts/).
    # Check both locations.

    feature_file = MODEL_DIR / "feature_columns.txt"

    if not feature_file.exists():
        feature_file = MODEL_DIR.parent / "feature_columns.txt"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature columns file not found in {MODEL_DIR} "
            f"or {MODEL_DIR.parent}"
        )

    with open(feature_file, "r") as f:
        FEATURE_COLS = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        f"[OK] Loaded {len(FEATURE_COLS)} feature columns "
        f"from {feature_file}"
    )

except Exception as e:
    raise Exception(
        f"Failed to load feature columns: {e}"
    )


# ============================================================
# FEATURE TRANSFORMATION CONSTANTS
# ============================================================

BINARY_MAP = {
    "gender": {
        "Female": 0,
        "Male": 1
    },

    "Partner": {
        "No": 0,
        "Yes": 1
    },

    "Dependents": {
        "No": 0,
        "Yes": 1
    },

    "PhoneService": {
        "No": 0,
        "Yes": 1
    },

    "PaperlessBilling": {
        "No": 0,
        "Yes": 1
    },
}


NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# SERVING TRANSFORMATION
# ============================================================

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature transformations used during training.
    """

    df = df.copy()

    # --------------------------------------------------------
    # STEP 1: Clean column names
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()


    # --------------------------------------------------------
    # STEP 2: Numeric type conversion
    # --------------------------------------------------------

    for c in NUMERIC_COLS:

        if c in df.columns:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            df[c] = df[c].fillna(0)


    # --------------------------------------------------------
    # STEP 3: Binary encoding
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # STEP 4: One-hot encode remaining categorical columns
    # --------------------------------------------------------

    obj_cols = list(
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    if obj_cols:

        df = pd.get_dummies(
            df,
            columns=obj_cols,
            drop_first=True
        )


    # --------------------------------------------------------
    # STEP 5: Convert boolean columns to integers
    # --------------------------------------------------------

    bool_cols = df.select_dtypes(
        include=["bool"]
    ).columns

    if len(bool_cols) > 0:

        df[bool_cols] = df[bool_cols].astype(int)


    # --------------------------------------------------------
    # STEP 6: Align with training feature schema
    # --------------------------------------------------------

    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict(input_dict: dict) -> str:
    """
    Generate a churn prediction for one customer.

    Returns:
        "Likely to churn"
        or
        "Not likely to churn"
    """

    # --------------------------------------------------------
    # STEP 1: Dictionary -> DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([input_dict])


    # --------------------------------------------------------
    # STEP 2: Apply training-compatible transformations
    # --------------------------------------------------------

    df_enc = _serve_transform(df)


    # --------------------------------------------------------
    # STEP 3: Model prediction
    # --------------------------------------------------------

    try:

        preds = model.predict(df_enc)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds

    except Exception as e:

        raise Exception(
            f"Model prediction failed: {e}"
        )


    # --------------------------------------------------------
    # STEP 4: Convert prediction to business output
    # --------------------------------------------------------

    if result == 1:

        return "Likely to churn"

    else:

        return "Not likely to churn"

