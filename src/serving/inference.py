import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib

def get_latest_model_and_encoder(experiment_name="Default"):
    client = MlflowClient()

    # Get experiment ID from name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # Get the most recent run for this experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["attributes.end_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in experiment.")

    run = runs[0]
    run_id = run.info.run_id

    # Load model & encoder
    model_uri = f"runs:/{run_id}/model"
    encoder_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="encoder.pkl"
    )

    model = mlflow.pyfunc.load_model(model_uri)
    encoder = joblib.load(encoder_path)

    return model, encoder


def predict(input_df: pd.DataFrame):
    model, encoder = get_latest_model_and_encoder()
    transformed = encoder.transform(input_df)
    preds = model.predict(transformed)
    return preds.tolist()
