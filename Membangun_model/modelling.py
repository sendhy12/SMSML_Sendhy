import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Enable autologging
mlflow.sklearn.autolog()

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

def train_model():
    """Train model with MLflow tracking"""
    
    # Set experiment
    mlflow.set_experiment("Basic_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        
        # MLflow will automatically log model and metrics
        print("Model training completed with autolog!")

if __name__ == "__main__":
    train_model()