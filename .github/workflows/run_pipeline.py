"""
Pipeline runner script for GitHub Actions
This script submits the MLOps pipeline to Azure ML
"""

import os
from azure.ai.ml import MLClient, Input, dsl, command
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

    # Get the data asset
    try:
        data_asset = ml_client.data.get(name="used-cars-data", version="1")
        print("Data asset found successfully")
    except Exception as e:
        print(f"Error getting data asset: {e}")
        return

    # Submit the pipeline (you would need to recreate the pipeline definition here
    # or import it from a separate module)
    print("Pipeline would be submitted here...")
    print("This is a placeholder for the actual pipeline submission logic")

if __name__ == "__main__":
    main()
