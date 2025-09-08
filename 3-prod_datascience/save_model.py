# Import KubeFlow Pipelines library
from kfp.dsl import (
    component,
    Input,
    Dataset,
    Metrics,
    Model,
    Artifact,
)


@component(base_image='python:3.11',
           packages_to_install=['pip==24.2',
                                'setuptools==74.1.3',
                                'boto3==1.36.12',
                                'model-registry'])

def push_to_model_registry(
    model_name: str,
    version: str,
    cluster_domain: str,
    s3_deployment_name: str,
    s3_region: str,
    author_name: str,
    torch_model: Input[Model],
    onnx_model: Input[Model],
    torch_metrics: Input[Metrics],
    scaler_model: Input[Model],
    label_encoder_model: Input[Model],
    training_dataset: Input[Dataset],
    hyperparameters: Input[Artifact],
):
    from model_registry import ModelRegistry
    from model_registry.utils import S3Params
    from model_registry.exceptions import StoreError
    import os

    # environment setup
    from os import environ
    environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    # Set up the model registry connection
    s3_deployment = s3_deployment_name
    model_registry_url = f"https://registry-rest.{cluster_domain}"
    minio_endpoint = f"https://{s3_deployment}.{cluster_domain}"

    # registry connection object
    registry = ModelRegistry(server_address=model_registry_url, port=443, author=author_name, is_secure=False)

    # Model details we want to register
    registered_model_name = model_name
    s3_model_bucket = "models"
    s3_model_prefix = f"{s3_model_bucket}/{registered_model_name}"
    version = version

    # remote S3 paths
    s3_region = s3_region,
    s3_onnx = f"{s3_model_prefix}/onnx/{registered_model_name}"
    s3_torch = f"{s3_model_prefix}/torch/{registered_model_name}"
    s3_artifacts = f"{s3_model_prefix}/artifacts/{registered_model_name}"
    s3_datasets = f"{s3_model_prefix}/datasets/{registered_model_name}"

    # upload parameters for s3 connections
    s3_upload_params_onnx = S3Params(
        bucket_name=os.environ.get('AWS_S3_BUCKET'),
        s3_prefix=f"{s3_onnx}/{version}",
    )
    s3_upload_params_torch = S3Params(
        bucket_name=os.environ.get('AWS_S3_BUCKET'),
        s3_prefix=f"{s3_torch}/{version}",
    )
    s3_upload_params_artifacts = S3Params(
        bucket_name=os.environ.get('AWS_S3_BUCKET'),
        s3_prefix=f"{s3_artifacts}/{version}",
    )
    s3_upload_params_datasets = S3Params(
        bucket_name=os.environ.get('AWS_S3_BUCKET'),
        s3_prefix=f"{s3_datasets}/{version}",
    )

    # artifact update function
    def update_artifact(model_name, model_version, new_uri, storage_path):
        artifact = registry.get_model_artifact(model_name, model_version)
        print(f"Got Artifact {artifact.name} with ID: {artifact.id}\n Current URI: {artifact.uri}\n Updating with URI: {new_uri}\n Current StoragePath: {artifact.storage_path}")
        artifact.uri = new_uri
        artifact.storage_path = storage_path
        registry.update(artifact)

    # upload function
    def register(model_name, data_path,
                 model_format_name, author, model_format_version,
                 model_version, storage_path, description,
                 metadata, upload_parms):
        try:
            # register onnx model
            registered_model = registry.upload_artifact_and_register_model(
                name=model_name,
                model_files_path=data_path,
                model_format_name=model_format_name,
                author=author,
                model_format_version=model_format_version,
                version=model_version,
                storage_path=storage_path,
                description=description,
                metadata=metadata,
                upload_params=upload_parms
            )
            print(f"'{model_name}' version '{model_version}'\n URL: https://rhods-dashboard-redhat-ods-applications.{cluster_domain}/modelRegistry/registry/registeredModels/1/versions/{registry.get_model_version(model_name, model_version).id}/details")
        except StoreError:
            stored_version = registry.get_model_version(registered_model_name, f"{version}-onnx")
            print(f"Model version {stored_version.name}-{stored_version.id} already exists: Updating URI...")
            new_uri = f"s3://{storage_path}?endpoint={minio_endpoint}&defaultRegion={s3_region}"
            update_artifact(model_name, model_version, new_uri, storage_path)

    # data to register in the model registry
    models = [
        {
            "model_name": registered_model_name,
            "data_path": onnx_model.path,
            "author": author_name,
            "model_format_name": "onnx",
            "model_format_version": "1",
            "model_version": f"{version}-onnx",
            "storage_path": f"{s3_model_bucket}/{s3_onnx}",
            "description": "Dense Neural Network trained on music data (ONNX)",
            "metadata": {
                        "version_info_uri": torch_metrics.path,
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_onnx
        },
        {
            "model_name": registered_model_name,
            "data_path": torch_model.path,
            "author": author_name,
            "model_format_name": "torch",
            "model_format_version": "1",
            "model_version": f"{version}-torch",
            "storage_path": f"{s3_model_bucket}/{s3_torch}",
            "description": "Dense Neural Network trained on music data (TORCH)",
            "metadata": {
                        "version_info_uri": torch_metrics.path,
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_torch
        },
        {
            "model_name": registered_model_name,
            "data_path": scaler_model.path,
            "author": author_name,
            "model_format_name": "pickle",
            "model_format_version": "1",
            "model_version": f"{version}-scaler",
            "storage_path": f"{s3_model_bucket}/{s3_artifacts}",
            "description": "Feature Scaler Model (Pickle)",
            "metadata": {
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_artifacts
        },
        {
            "model_name": registered_model_name,
            "data_path": label_encoder_model.path,
            "author": author_name,
            "model_format_name": "pickle",
            "model_format_version": "1",
            "model_version": f"{version}-scaler",
            "storage_path": f"{s3_model_bucket}/{s3_artifacts}",
            "description": "Label Encoder Model (Pickle)",
            "metadata": {
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_artifacts
        },
        {
            "model_name": registered_model_name,
            "data_path": training_dataset.path,
            "author": author_name,
            "model_format_name": "parquet",
            "model_format_version": "1",
            "model_version": f"{version}-dataset",
            "storage_path": f"{s3_model_bucket}/{s3_datasets}",
            "description": "Training Dataset (parquet)",
            "metadata": {
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_datasets
        },
        {
            "model_name": registered_model_name,
            "data_path": hyperparameters.path,
            "author": author_name,
            "model_format_name": "json",
            "model_format_version": "1",
            "model_version": f"{version}-hyperparameters",
            "storage_path": f"{s3_model_bucket}/{s3_datasets}",
            "description": "Training Hyperparameters",
            "metadata": {
                        "dataset": training_dataset.path,
                        "training_parameters": hyperparameters.path,
                        "license": "apache-2.0"
                    },
            "upload_parms": s3_upload_params_datasets
        }
    ]

    # register models
    for model in models:
        print(f"Registering: {model.get('model_version')}...")
        register(model_name=model.get('model_name'),
                 data_path=model.get('data_path'),
                 model_format_name=model.get('model_format_name'),
                 author=model.get('author'),
                 model_format_version=model.get('model_format_version'),
                 model_version=model.get('model_version'),
                 storage_path=model.get('storage_path'),
                 description=model.get('description'),
                 metadata=model.get('metadata'),
                 upload_parms=model.get('upload_parms'))

    print("Model registered successfully")