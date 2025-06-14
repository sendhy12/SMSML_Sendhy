# automate_sendhy.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load raw dataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        return pd.read_csv(file_path)
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_clean = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
        # Categorical columns: fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            
        return df_clean
    
    def encode_categorical(self, df, target_column):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col != target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                
        return df_encoded
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_data(self, file_path, target_column, test_size=0.2):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        print("Handling missing values...")
        df_clean = self.handle_missing_values(df)
        
        print("Encoding categorical variables...")
        df_encoded = self.encode_categorical(df_clean, target_column)
        
        print("Splitting data...")
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print("Scaling features...")
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return X_train_df, X_test_df, y_train, y_test
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir):
        """Save preprocessed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save preprocessing objects
        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{output_dir}/label_encoders.pkl", 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"Preprocessed data saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        
        # Cek berbagai kemungkinan lokasi file dataset
        possible_paths = [
            "heart.csv",  # File ada di root directory berdasarkan struktur yang ditunjukkan
            "data/heart.csv", 
            "dataset/heart.csv",
            "Eksperimen_SML_Sendhy/heart.csv",
            "D:\laragon\www\Eksperimen_SML_Sendhy\heart.csv",
            os.path.join("Eksperimen_SML_Sendhy", "heart.csv")
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            print("Dataset file tidak ditemukan di lokasi manapun.")
            print("File yang tersedia:")
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith('.csv'):
                        print(f"  {os.path.join(root, file)}")
            exit(1)
            
        print(f"Using dataset: {dataset_path}")
        target_column = "target"
        output_directory = "./dataset_preprocessing"
        
        # Run preprocessing
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            dataset_path, target_column
        )
        
        # Save results
        preprocessor.save_preprocessed_data(
            X_train, X_test, y_train, y_test, output_directory
        )
        
        print("Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        exit(1)