import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X, y):
    """
    Tunes a RandomForest model using Optuna.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 2, 20)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best Params:", study.best_params)
    return study.best_params
