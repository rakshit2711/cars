"""
Model Registration Script for Used Cars Price Prediction
This script registers the best trained model in MLflow registry.
"""

import argparse
import os
import mlflow
import mlflow.sklearn
import joblib

def main():
    """Main function to execute model registration"""

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(f"Loading model from: {args.model}")

    # Load the model
    model_path = os.path.join(args.model, "model.pkl")
    model = joblib.load(model_path)

    # Load preprocessors
    preprocessors_path = os.path.join(args.model, "preprocessors.pkl")
    preprocessors = joblib.load(preprocessors_path)

    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")

    # Register the model in MLflow
    print("Registering model in MLflow...")

    # Log the model with all artifacts
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
        registered_model_name="used_cars_price_prediction_model",
        signature=None,  # You can add model signature here for better tracking
        input_example=None  # You can add input example here
    )

    # Log preprocessors as artifacts
    mlflow.log_artifact(preprocessors_path, "preprocessors")

    print("Model registered successfully in MLflow!")
    print("Model name: used_cars_price_prediction_model")
    print("Artifact path: random_forest_price_regressor")

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
