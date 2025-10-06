"""
Pipeline runner script for GitHub Actions
This script submits the MLOps pipeline to Azure ML
"""

import os
import sys
from pathlib import Path

try:
    from azure.ai.ml import MLClient, Input, dsl, command
    from azure.identity import DefaultAzureCredential
    AZURE_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Azure libraries not available: {e}")
    AZURE_LIBS_AVAILABLE = False

def create_pipeline_components():
    """Create pipeline components"""

    # Data preparation component
    data_prep_job = command(
        inputs=dict(
            data=Input(type="uri_file"),
            test_train_ratio=0.2,
        ),
        outputs=dict(
            train_data={"type": "uri_folder", "mode": "rw_mount"},
            test_data={"type": "uri_folder", "mode": "rw_mount"},
        ),
        code="./src",
        command="python data_prep.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}",
        environment="machine_learning_E2E@latest",
        compute="cars",
        display_name="data_preparation",
        description="Data preparation for used cars price prediction",
    )

    # Training component
    train_job = command(
        inputs=dict(
            train_data=Input(type="uri_folder"),
            test_data=Input(type="uri_folder"),
            n_estimators=100,
            max_depth=-1,
        ),
        outputs=dict(
            model_output={"type": "uri_folder", "mode": "rw_mount"},
        ),
        code="./src",
        command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --model_output ${{outputs.model_output}}",
        environment="machine_learning_E2E@latest",
        compute="cars",
        display_name="train_model",
        description="Train Random Forest model for used cars price prediction",
    )

    # Registration component
    register_job = command(
        inputs=dict(
            model=Input(type="uri_folder"),
        ),
        code="./src",
        command="python model_register.py --model ${{inputs.model}}",
        environment="machine_learning_E2E@latest",
        compute="cars",
        display_name="register_model",
        description="Register the trained model in MLflow registry",
    )

    return data_prep_job, train_job, register_job

@dsl.pipeline(
    compute="cars",
    description="End-to-End MLOps Pipeline for Used Cars Price Prediction",
)
def used_cars_pipeline(
    pipeline_data,
    test_train_ratio=0.2,
    n_estimators=100,
    max_depth=-1,
):
    """
    End-to-end pipeline for used cars price prediction
    """
    if not AZURE_LIBS_AVAILABLE:
        raise ImportError("Azure ML libraries are required to run this pipeline")

    data_prep_job, train_job, register_job = create_pipeline_components()

    # Step 1: Data Preparation
    data_prep_step = data_prep_job(
        data=pipeline_data,
        test_train_ratio=test_train_ratio,
    )

    # Step 2: Model Training
    train_step = train_job(
        train_data=data_prep_step.outputs.train_data,
        test_data=data_prep_step.outputs.test_data,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Step 3: Model Registration
    register_step = register_job(
        model=train_step.outputs.model_output,
    )

    return {
        "train_data": data_prep_step.outputs.train_data,
        "test_data": data_prep_step.outputs.test_data,
        "model": train_step.outputs.model_output,
    }

def main():
    """Main function to execute the pipeline"""

    if not AZURE_LIBS_AVAILABLE:
        print("Azure ML libraries are not available. Skipping pipeline execution.")
        print("This would normally submit the pipeline to Azure ML.")
        return

    try:
        # Get environment variables
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        resource_group = os.environ.get("AZURE_RESOURCE_GROUP") 
        workspace_name = os.environ.get("AZURE_ML_WORKSPACE")

        if not all([subscription_id, resource_group, workspace_name]):
            print("Missing required environment variables:")
            print(f"AZURE_SUBSCRIPTION_ID: {subscription_id}")
            print(f"AZURE_RESOURCE_GROUP: {resource_group}")
            print(f"AZURE_ML_WORKSPACE: {workspace_name}")
            return

        # Initialize ML Client
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )

        print(f"Connected to workspace: {workspace_name}")

        # Get the data asset
        try:
            data_asset = ml_client.data.get(name="used-cars-data", version="1")
            print("Data asset found successfully")
        except Exception as e:
            print(f"Error getting data asset: {e}")
            print("Creating data asset might be required first")
            return

        # Create and submit pipeline
        pipeline = used_cars_pipeline(
            pipeline_data=Input(type="uri_file", path=data_asset.path),
            test_train_ratio=0.2,
            n_estimators=100,
            max_depth=-1,
        )

        # Submit the pipeline job
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline,
            experiment_name="used-cars-price-prediction-ci"
        )

        print(f"Pipeline job submitted with ID: {pipeline_job.name}")
        print(f"Pipeline status: {pipeline_job.status}")
        print(f"Pipeline URL: {pipeline_job.services.get('Studio', {}).get('endpoint', 'N/A')}")

    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("This might be expected in a CI/CD environment without proper Azure credentials.")
        return

if __name__ == "__main__":
    main()
