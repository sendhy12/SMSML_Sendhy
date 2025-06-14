import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/sendhy12/modelling.mlflow/'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sendhy12'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'feed3e8ecbba6b109d64fecfb04eb1fb648d230d'


import os
import pandas as pd

def load_data():
    """Load preprocessed data from Eksperimen_SML_Sendhy/preprocessing/preprocessed-data"""
    # Ambil direktori file ini
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Bangun path ke folder preprocessed-data
    data_dir = os.path.join(base_dir, "..", "Eksperimen_SML_Sendhy", "preprocessing", "preprocessed-data")

    # Baca dataset
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    return X_train, X_test, y_train, y_test

def train_model_with_tuning():
    """Train model with hyperparameter tuning and manual logging"""
    
    # Set experiment
    mlflow.set_experiment("Skilled_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Manual logging (same as autolog)
        mlflow.log_param("n_estimators", best_model.n_estimators)
        mlflow.log_param("max_depth", best_model.max_depth)
        mlflow.log_param("min_samples_split", best_model.min_samples_split)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log best parameters
        mlflow.log_params(grid_search.best_params_)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    train_model_with_tuning()