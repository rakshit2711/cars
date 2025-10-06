"""
Pipeline runner script for GitHub Actions
This script submits the MLOps pipeline to Azure ML
"""

import os
import argparse
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

def run_data_prep(ml_client):
    """Run data preparation step"""
    print("Running data preparation step...")
    
    # List all available compute resources to debug
    print("Checking available compute resources...")
    try:
        computes = ml_client.compute.list()
        for compute in computes:
            print(f"Found compute: {compute.name} (Type: {compute.type}, State: {compute.provisioning_state})")
    except Exception as e:
        print(f"Error listing compute resources: {e}")
    
    # Try different compute targets, prioritizing compute clusters
    
    # Get available compute clusters (same logic as data prep)
    compute_targets = [
        "cars",
        "cars-github"
    ]
       
    compute_name = None
    
    for target in compute_targets:
        try:
            compute = ml_client.compute.get(target)
            # Check if it's a compute cluster (not instance)
            if compute.type == "amlcompute":
                print(f"Using compute cluster: {target}")
                compute_name = target
                break
            else:
                print(f"Skipping {target} (Type: {compute.type}) - need compute cluster")
        except Exception as e:
            print(f"Compute {target} not available: {e}")
            continue
    
    if not compute_name:
        print("No suitable compute cluster found. Creating a new one...")
        compute_name = "github-actions-cluster"
        try:
            compute = AmlCompute(
                name=compute_name,
                type="amlcompute",
                size="Standard_DS3_v2",
                min_instances=0,
                max_instances=2,
                idle_time_before_scale_down=120,
                tier="Dedicated",
            )
            print(f"Creating compute cluster: {compute_name}")
            operation = ml_client.compute.begin_create_or_update(compute)
            print("Waiting for compute creation to complete...")
            operation.wait()
            print(f"Successfully created compute cluster: {compute_name}")
        except Exception as e:
            print(f"Failed to create compute cluster: {e}")
            print("This might be due to quota limits or permissions.")
            print("Please create a compute cluster manually in Azure ML Studio.")
            raise
    
    # Get the data asset
    try:
        data_asset = ml_client.data.get(name="used-cars-data", version="1")
        print("Data asset found successfully")
    except Exception as e:
        print(f"Error getting data asset: {e}")
        return
    
    # Define the data preparation job
    data_prep_job = command(
        inputs=dict(
            data=Input(type="uri_file", path=data_asset.path),
            test_train_ratio=0.2,
        ),
        outputs=dict(
            train_data=Output(type="uri_folder", mode="rw_mount"),
            test_data=Output(type="uri_folder", mode="rw_mount"),
        ),
        code="./src",
        command="python data_prep.py --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}",
        environment="machine_learning_E2E@latest",
        compute=compute_name,
        display_name="data_preparation",
        description="Data preparation for used cars price prediction",
    )
    
    # Submit the job
    returned_job = ml_client.jobs.create_or_update(data_prep_job)
    print(f"Data prep job submitted. Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    return returned_job.name

def run_training(ml_client):
    """Run model training step"""
    print("Running model training step...")
    
    # Get available compute clusters (same logic as data prep)
    compute_targets = [
        "cars",
        "cars-github"
    ]
    
    compute_name = None
    
    for target in compute_targets:
        try:
            compute = ml_client.compute.get(target)
            if compute.type == "amlcompute":
                print(f"Using compute cluster: {target}")
                compute_name = target
                break
            else:
                print(f"Skipping {target} (Type: {compute.type}) - need compute cluster")
        except Exception as e:
            print(f"Compute {target} not available: {e}")
            continue
    
    if not compute_name:
        print("No suitable compute cluster found for training.")
        raise Exception("Please ensure a compute cluster is available")
    
    # Try to get the latest data preparation outputs or use default paths
    try:
        # You may need to adjust these paths based on your actual data asset names
        train_data_path = "azureml://datastores/workspaceblobstore/paths/LocalUpload/train/"
        test_data_path = "azureml://datastores/workspaceblobstore/paths/LocalUpload/test/"
    except:
        # Fallback to default paths
        train_data_path = "azureml://datastores/workspaceblobstore/paths/train/"
        test_data_path = "azureml://datastores/workspaceblobstore/paths/test/"
    
    train_job = command(
        inputs=dict(
            train_data=Input(type="uri_folder", path=train_data_path),
            test_data=Input(type="uri_folder", path=test_data_path),
            n_estimators=100,
            max_depth=-1,
        ),
        outputs=dict(
            model_output=Output(type="uri_folder", mode="rw_mount"),
        ),
        code="./src",
        command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --n_estimators ${{inputs.n_estimators}} --max_depth ${{inputs.max_depth}} --model_output ${{outputs.model_output}}",
        environment="machine_learning_E2E@latest",
        compute=compute_name,
        display_name="train_model",
        description="Train Random Forest model for used cars price prediction",
    )

    # Submit the job
    returned_job = ml_client.jobs.create_or_update(train_job)
    print(f"Training job submitted. Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    return returned_job.name

def run_registration(ml_client):
    """Run model registration step"""
    print("Running model registration step...")
    
    # Get available compute clusters (same logic as other steps)
    
    # Get available compute clusters (same logic as data prep)
    compute_targets = [
        "cars",
        "cars-github"
    ]
    
    compute_name = None
    
    for target in compute_targets:
        try:
            compute = ml_client.compute.get(target)
            if compute.type == "amlcompute":
                print(f"Using compute cluster: {target}")
                compute_name = target
                break
            else:
                print(f"Skipping {target} (Type: {compute.type}) - need compute cluster")
        except Exception as e:
            print(f"Compute {target} not available: {e}")
            continue
    
    if not compute_name:
        print("No suitable compute cluster found for registration.")
        raise Exception("Please ensure a compute cluster is available")
    
    # Try to get the latest model output or use default path
    try:
        model_path = "azureml://datastores/workspaceblobstore/paths/LocalUpload/model/"
    except:
        model_path = "azureml://datastores/workspaceblobstore/paths/model/"
    
    register_job = command(
        inputs=dict(
            model=Input(type="uri_folder", path=model_path),
        ),
        code="./src",
        command="python model_register.py --model ${{inputs.model}}",
        environment="machine_learning_E2E@latest",
        compute=compute_name,
        display_name="register_model",
        description="Register the trained model in MLflow registry",
    )

    # Submit the job
    returned_job = ml_client.jobs.create_or_update(register_job)
    print(f"Registration job submitted. Job name: {returned_job.name}")
    print(f"Job status: {returned_job.status}")
    return returned_job.name

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MLOps pipeline steps")
    parser.add_argument("--step", type=str, required=True, 
                       choices=["data_prep", "training", "registration"],
                       help="Pipeline step to run")
    args = parser.parse_args()
    
    print(f"Starting MLOps pipeline step: {args.step}")
    
    # Get environment variables
    try:
        subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
        resource_group = os.environ["AZURE_RESOURCE_GROUP"]
        workspace_name = os.environ["AZURE_ML_WORKSPACE"]
        print(f"Environment variables loaded successfully")
        print(f"Workspace: {workspace_name}")
        print(f"Resource Group: {resource_group}")
    except KeyError as e:
        print(f"Missing required environment variable: {e}")
        return
    
    # Initialize ML Client
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        print(f"Successfully connected to Azure ML workspace: {workspace_name}")
    except Exception as e:
        print(f"Failed to connect to Azure ML workspace: {e}")
        return
    
    # Run the appropriate step
    try:
        if args.step == "data_prep":
            run_data_prep(ml_client)
        elif args.step == "training":
            run_training(ml_client)
        elif args.step == "registration":
            run_registration(ml_client)
        else:
            print(f"Unknown step: {args.step}")
            return
        
        print(f"Successfully completed step: {args.step}")
        
    except Exception as e:
        print(f"Error running step {args.step}: {e}")
        raise

if __name__ == "__main__":
    main()
