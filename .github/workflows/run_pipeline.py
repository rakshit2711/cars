"""
Pipeline runner script for GitHub Actions
This script submits the MLOps pipeline to Azure ML
"""

import os
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

def main():
    # Get environment variables
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"] 
    workspace_name = os.environ["AZURE_ML_WORKSPACE"]

    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    print(f"Connected to workspace: {workspace_name}")

    # Define the command job to run train.py
    train_job = command(
    inputs=dict(
        train_data=Input(type="uri_folder"),
        test_data=Input(type="uri_folder"),
        n_estimators=100,
        max_depth=-1,  # -1 represents None
    ),
    outputs=dict(
        model_output=Output(type="uri_folder", mode="rw_mount"),
    ),
    code="./src",  # location of source code
    command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --model_output ${{outputs.model_output}}",
    environment="machine_learning_E2E@latest",
    compute="cars",
    display_name="train_model",
    description="Train Random Forest model for used cars price prediction",
)

    # Submit the job
    returned_job = ml_client.jobs.create_or_update(train_job)
    print(f"Job submitted. Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")

if __name__ == "__main__":
    main()

