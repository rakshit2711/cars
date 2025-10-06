"""
Model Training Script for Used Cars Price Prediction
This script handles model training with hyperparameter tuning and evaluation.
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib

def preprocess_features(df, label_encoders=None, scaler=None, is_training=True):
    """Preprocess features for training or prediction"""

    # Separate features and target
    if 'price' in df.columns:
        X = df.drop(['price'], axis=1)
        y = df['price']
    else:
        X = df.copy()
        y = None

    # Handle categorical variables
    categorical_cols = ['segment']

    if is_training:
        label_encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        # Scale numerical features
        numerical_cols = ['kilometers_driven', 'mileage', 'engine', 'power', 'seats']
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    else:
        # Apply existing encoders and scaler
        for col in categorical_cols:
            if col in X.columns and col in label_encoders:
                X[col] = label_encoders[col].transform(X[col].astype(str))

        numerical_cols = ['kilometers_driven', 'mileage', 'engine', 'power', 'seats']
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    return X, y, label_encoders, scaler

def main():
    """Main function to execute model training"""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--n_estimators", type=int, required=False, default=100)
    parser.add_argument("--max_depth", type=int, required=False, default=None)
    parser.add_argument("--model_output", type=str, help="path to model file")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))

    print("Loading testing data...")
    test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # Preprocess the data
    print("Preprocessing training data...")
    X_train, y_train, label_encoders, scaler = preprocess_features(train_df, is_training=True)

    print("Preprocessing testing data...")
    X_test, y_test, _, _ = preprocess_features(test_df, label_encoders, scaler, is_training=False)

    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")

    # Initialize and train the model
    print("Training Random Forest model...")
    max_depth = args.max_depth if args.max_depth != -1 else None

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    # Log metrics
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("test_mae", test_mae)

    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")

    # Create the output directory
    os.makedirs(args.model_output, exist_ok=True)

    # Save the model and preprocessors
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)

    preprocessors_path = os.path.join(args.model_output, "preprocessors.pkl")
    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    joblib.dump(preprocessors, preprocessors_path)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=None
    )

    print("Model training completed successfully!")

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
