# Import KubeFlow Pipelines library
import kfp

# Import objects from the DSL library
from kfp.dsl import pipeline
from kfp import kubernetes

# Component imports
from fetch_data import fetch_data
from data_validation import validate_data
from train_model import train_torch_model
from convert_model import convert_torch_to_onnx
from evaluate_model import evaluate_torch_model_performance, save_torch_metrics, validate_onnx_model
from save_model import push_to_model_registry

# Pipeline definition

# name of the data connection that points to the s3 model storage bucket
data_connection_secret_name = 's3-models'
network_settings_secret = 'network-settings'


# Create pipeline
@pipeline(
  name='jukebox-training-pipeline',
  description='Jukebox Song Prediction Model Training Pipeline'
)
def training_pipeline(hyperparameters: dict,
                      model_name: str,
                      version: str,
                      s3_deployment_name: str,
                      s3_region: str,
                      author_name: str,
                      cluster_domain: str,
                      model_storage_pvc: str,
                      prod_flag: bool):
    # Fetch Data from GitHub
    fetch_task = fetch_data(version=version)
    kubernetes.use_secret_as_env(
        fetch_task,
        secret_name=network_settings_secret,
        secret_key_to_env={
            "HTTP_PROXY": "HTTP_PROXY",
            "HTTPS_PROXY": "HTTPS_PROXY",
        },
    )

    # Validate Data
    data_validation_task = validate_data(version=version, dataset=fetch_task.outputs["dataset"])

    # Train Keras model
    training_task = train_torch_model(
        version=version,
        train_dataset=fetch_task.outputs["dataset"],
        hyperparameters=hyperparameters,
    )
    training_task.set_memory_limit("8G")

    # Convert Torch model to ONNX
    convert_task = convert_torch_to_onnx(version=version, torch_model=training_task.outputs["trained_model"],
                                        hyperparameters=training_task.outputs["training_parameters"])

    # Evaluate Trained Torch model performance
    model_evaluation_task = evaluate_torch_model_performance(
        torch_model=training_task.outputs["trained_model"],
        hyperparameters=training_task.outputs["training_parameters"],
        train_dataset=fetch_task.outputs["dataset"],
        model_name=model_name,
        cluster_domain=cluster_domain,
        prod_flag=prod_flag,
        version=version, # Add version to force a rerun of this step every new version
    )
    model_evaluation_task.set_memory_limit("8G")
    kubernetes.use_field_path_as_env(
        model_evaluation_task,
        env_name='NAMESPACE',
        field_path='metadata.namespace'
    )

    # save performance metrics
    model_metrics_taks = save_torch_metrics(
        model_test_information=model_evaluation_task.outputs["testing_artifact"],
        model_name=model_name,
        cluster_domain=cluster_domain,
        version=version,
        prod_flag=prod_flag,
    )
    kubernetes.use_field_path_as_env(
        model_metrics_taks,
        env_name='NAMESPACE',
        field_path='metadata.namespace'
    )

    # Validate that the Torch -> ONNX conversion was successful
    model_validation_task = validate_onnx_model(
        version=version,
        onnx_model=convert_task.outputs["onnx_model"],
        training_data=fetch_task.outputs["dataset"],
        hyperparameters=training_task.outputs["training_parameters"],
        torch_metrics=model_evaluation_task.outputs["testing_artifact"]
    )
    model_validation_task.set_memory_limit("8G")

    # Register model to the Model Registry
    register_model_task = push_to_model_registry(
        model_name=model_name,
        version=version,
        cluster_domain=cluster_domain,
        s3_deployment_name=s3_deployment_name,
        s3_region=s3_region,
        author_name=author_name,
        torch_model=training_task.outputs["trained_model"],
        onnx_model=convert_task.outputs["onnx_model"],
        torch_metrics=model_evaluation_task.outputs["testing_artifact"],
        scaler_model=training_task.outputs["minmax_scaler_model"],
        label_encoder_model=training_task.outputs["label_encoder_model"],
        training_dataset=fetch_task.outputs["dataset"],
        hyperparameters=training_task.outputs["training_parameters"],
    )
    kubernetes.use_secret_as_env(
        register_model_task,
        secret_name=data_connection_secret_name,
        secret_key_to_env={
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
        },
    )
    kubernetes.use_secret_as_env(
        register_model_task,
        secret_name=network_settings_secret,
        secret_key_to_env={
            "HTTP_PROXY": "HTTP_PROXY",
            "HTTPS_PROXY": "HTTPS_PROXY",
        },
    )


# start pipeline
if __name__ == '__main__':
    metadata = {
        "hyperparameters": {
            "epochs": 2,
            "base_size": 32,
            "learning_rate": 1e-3,
            "batch_size": 128,
        },
        "model_name": "jukebox",
        "version": "0.0.6",
        "s3_deployment_name": "minio-s3-s3-minio-dev",
        "s3_region": "us",
        "author_name": "DevOps Team",
        "cluster_domain": "apps.xxx-yyy.local",
        "model_storage_pvc": "jukebox-model-pvc",
        "prod_flag": False
    }

    namespace_file_path =\
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    with open(namespace_file_path, 'r') as namespace_file:
        namespace = namespace_file.read()

    kubeflow_endpoint =\
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'

    sa_token_file_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(sa_token_file_path, 'r') as token_file:
        bearer_token = token_file.read()

    ssl_ca_cert =\
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert
    )

    client.create_run_from_pipeline_func(
        training_pipeline,
        arguments=metadata,
        experiment_name="jukebox-training-pipeline",
        enable_caching=True
    )
