import os
import pandas as pd
import mlflow.xgboost  # use MLflow’s loader for XGBoost models

# Path to bundled model inside repo
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), 
    "model", 
    "2ac205f95a264d49b964ab362fe5f4e6", 
    "artifacts", 
    "model"
)

# Load model once at startup
model = mlflow.xgboost.load_model(MODEL_DIR)

def predict(input_df: pd.DataFrame):
    """
    Takes input dataframe, applies model, and returns prediction.
    """
    preds = model.predict(input_df)
    return preds.tolist()
