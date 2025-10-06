"""
Data Preparation Script for Used Cars Price Prediction
This script handles data loading, cleaning, and splitting for the MLOps pipeline.
"""

import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def main():
    """Main function to execute data preparation"""

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print("Input Data:", args.data)
    print("Test/Train Ratio:", args.test_train_ratio)

    # Read the data
    print("Reading data...")
    all_data = pd.read_csv(args.data)

    print(f"Dataset shape: {all_data.shape}")
    print("Dataset info:")
    print(all_data.info())
    print("\nDataset description:")
    print(all_data.describe())

    # Check for missing values
    print("\nMissing values:")
    print(all_data.isnull().sum())

    # Data preprocessing
    print("\nStarting data preprocessing...")

    # Handle missing values if any
    if all_data.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # For numeric columns, fill with median
        numeric_cols = all_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col].fillna(all_data[col].median(), inplace=True)

        # For categorical columns, fill with mode
        categorical_cols = all_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if all_data[col].isnull().sum() > 0:
                all_data[col].fillna(all_data[col].mode()[0], inplace=True)

    # Remove any duplicates
    initial_rows = len(all_data)
    all_data = all_data.drop_duplicates()
    print(f"Removed {initial_rows - len(all_data)} duplicate rows")

    # Convert column names to lowercase for consistency
    all_data.columns = all_data.columns.str.lower()

    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(
        all_data,
        test_size=args.test_train_ratio,
        random_state=42,
        stratify=all_data['segment']  # Stratify by segment to maintain distribution
    )

    # Create the output directories
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Save the train and test sets
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Log key metrics
    mlflow.log_metric("total_samples", len(all_data))
    mlflow.log_metric("train_samples", len(train_df))
    mlflow.log_metric("test_samples", len(test_df))
    mlflow.log_metric("train_test_ratio", args.test_train_ratio)

    print(f"Total samples: {len(all_data)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Test ratio: {args.test_train_ratio}")

    # Log data distribution insights
    print("\nSegment distribution in training data:")
    segment_dist = train_df['segment'].value_counts()
    print(segment_dist)

    for segment, count in segment_dist.items():
        mlflow.log_metric(f"train_{segment.replace(' ', '_')}_count", count)

    print("\nData preparation completed successfully!")

    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()
