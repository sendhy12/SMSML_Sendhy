import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def train_model():
    # Set tracking URI if available
    if os.getenv('MLFLOW_TRACKING_URI'):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    
    mlflow.set_experiment("Docker_ML_Experiment")
    
    with mlflow.start_run():
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv').values.ravel()
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv').values.ravel()
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained with accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()