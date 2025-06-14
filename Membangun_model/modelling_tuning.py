import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Setup MLflow tracking ke DagsHub (gunakan env vars atau secrets di GitHub Actions)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow.set_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

def load_data():
    """Load preprocessed data dari Eksperimen_SML_Sendhy/preprocessing/preprocessed-data"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "Eksperimen_SML_Sendhy", "preprocessing", "preprocessed-data")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test

def train_model_with_tuning():
    """Train model with hyperparameter tuning and manual logging"""

    # Jangan set experiment (DagsHub tidak support endpoint ini)
    # mlflow.set_experiment("Skilled_ML_Experiment")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_data()

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Logging ke MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(best_model, "model")

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    train_model_with_tuning()
