# Import objects from KubeFlow Pipelines DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Metrics,
    Model,
    Artifact,
)


@component(base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
           packages_to_install=["pandas==2.2.3",
                                "dask[dataframe]==2024.8.0",
                                "scikit-learn==1.6.1"])
def evaluate_torch_model_performance(
    torch_model: Input[Model],
    hyperparameters: Input[Artifact],
    train_dataset: Input[Dataset],
    model_name: str,
    cluster_domain: str,
    version: str,
    prod_flag: bool,
    testing_artifact: Output[Artifact]
):
    import torch
    import torch.cuda as tc
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    # data libraries
    import pandas as pd

    # sklearn
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    # torch custom dataset for this usecase
    class JukeboxDataset(Dataset):
        def __init__(self, dataset_in, features_columns, label_column=None):
            self.features_columns = features_columns
            self.label_column = label_column

            # read datasets and get relevant features inside
            self.dataset_in = pd.read_parquet(dataset_in)
            self.merged_dataset = self.dataset_in[self.features_columns]

            # prepare labels if needed
            if self.label_column:
                self.labels_data = self.dataset_in[self.label_column]
                # encode labels
                self.label_encoder = LabelEncoder()
                encoded_labels = torch.from_numpy(self.label_encoder.fit_transform(self.labels_data))
                self.labels = F.one_hot(encoded_labels)

            # normalize dataset between [-1,1]
            # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier 
            # for the model to learn than random (and potentially large) values.
            # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global 
            # distribution of variables (which is influenced by the test set) into the training set.
            self.minmaxscaler = MinMaxScaler((-1, 1))
            self.merged_dataset = pd.DataFrame(self.minmaxscaler.fit_transform(self.merged_dataset), index=self.merged_dataset.index, columns=self.merged_dataset.columns)

        def __len__(self):
            return len(self.merged_dataset)

        def __getitem__(self, idx):
            """
            Retrieves a single row from the dataset file.
            """
            dp = self.merged_dataset.iloc[idx]

            if self.label_column:
                label = self.labels[idx]
                return torch.tensor(dp.values).float(), label.float()
            else:
                return torch.tensor(dp.values).float()  # For unsupervised learning

    # detect device
    device = "cpu"
    if tc.is_available():
        device = "cuda"

    print(f"DEVICE:\n Training on {device}")

    # Import torch specific libraries
    from torch.nn import Module, Sequential, ReLU, Linear
    from collections import OrderedDict

    """ Define the Neural Network Architecture """

    # build the neural network that will predict the country out of selected features
    class CountryPredictorNetwork(Module):
        def __init__(self, n_inputs, hidden_len, n_outputs):
            super().__init__()
            self.predictor_network = Sequential(OrderedDict([
                ("input", Linear(n_inputs, hidden_len)),
                ("input_activation", ReLU()),
                ("linear_0", Linear(hidden_len, hidden_len * 2)),
                ("activation_0", ReLU()),
                ("linear_1", Linear(hidden_len * 2, hidden_len * 4)),
                ("activation_1", ReLU()),
                ("linear_2", Linear(hidden_len * 4, hidden_len * 8)),
                ("activation_2", ReLU()),
                ("output", Linear(hidden_len * 8, n_outputs)),
            ]))

        def forward(self, input_sample):
            return self.predictor_network(input_sample)

    # load parameters
    import json
    parms = {}
    with open(hyperparameters.path, "r") as p:
        parms = json.load(p)

    # instantiate a model object
    n_feats = parms.get("n_feats")
    base_size = parms.get("base_size")
    n_classes = parms.get("n_classes")
    # create a model instance
    model = CountryPredictorNetwork(n_feats, base_size, n_classes)

    # load pretrained weights from checkpoint
    model.load_state_dict(torch.load(torch_model.path))

    # load testing dataset
    dataset = JukeboxDataset(train_dataset.path,
                             parms.get("dataset_features"),
                             parms.get("label_feature"))
    test_data = torch.utils.data.Subset(dataset, parms.get("test_indices"))

    """ Test Model Performance """
    # model test function
    def test_loop(model, dataloader, device="cpu") -> float:
        model.to(device)
        model.eval()
        # loop over test data
        correct_guesses = 0
        total_guesses = 0
        with torch.no_grad():
            for b, (point, label) in enumerate(dataloader):
                # make predictions
                y_hat = model(point.to(device))
                # calculate softmax (probability distribution across classes)
                y_hat_prob = torch.softmax(y_hat, dim=0)
                _, category = torch.max(y_hat_prob, dim=0)

                # calculate accuracy
                total_guesses += label.size(0)
                correct_guesses += (category == label).sum().item()

        print(f"Model Accuracy: {100*correct_guesses/total_guesses:.2f}%")
        return float((correct_guesses/total_guesses))

    # load validation data
    batch_size = parms.get("batch_size")
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # validate model
    accuracy = test_loop(model, test_dataloader, device)

    # save testing data
    test_results = {
        "model_name": model_name,
        "model_version": version,
        "accuracy": accuracy,
        "num_test_samples": len(test_data.indices)
    }
    import json
    testing_artifact.path += ".json"
    with open(testing_artifact.path, "w") as td:
        json.dump(test_results, td)


@component(base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
           packages_to_install=["pandas==2.2.3",
                                "dask[dataframe]==2024.8.0",
                                "scikit-learn==1.6.1",
                                "model-registry==0.2.10"])
def save_torch_metrics(
    model_test_information: Input[Artifact],
    model_name: str,
    cluster_domain: str,
    version: str,
    prod_flag: bool,
    metrics: Output[Metrics]
):
    import json
    training_metrics = {}
    with open(model_test_information.path, "r") as f:
        training_metrics = json.load(f)

    accuracy = training_metrics.get("accuracy")

    # Get the previous models properties from the Model Registry
    from os import environ
    if prod_flag:
        namespace = environ.get("NAMESPACE").split("-")[0]+"-prod"
    else:
        namespace = environ.get("NAMESPACE").split("-")[0]

    from model_registry import ModelRegistry
    from model_registry.exceptions import StoreError
    environ["KF_PIPELINES_SA_TOKEN_PATH"] = "/var/run/secrets/kubernetes.io/serviceaccount/token" # Hotfix to access the endpoint
    registry = ModelRegistry(server_address=f"https://registry-rest.{cluster_domain}", port=443, author="someone", is_secure=False)
    previous_model_properties = {}

    # Wrap with try except to see if the model exists in the registry
    try:
        # Get the latest models properties if no model is in production
        for v in registry.get_model_versions(model_name).order_by_id().descending():
            if not previous_model_properties:
                previous_model_properties = registry.get_model_versions(model_name).order_by_id().descending().next_item().custom_properties
            elif "prod" in v.custom_properties and v.custom_properties["prod"]:
                previous_model_properties = v.custom_properties
                break
    except StoreError:
        pass

    if "accuracy" not in previous_model_properties:
        previous_model_properties["accuracy"] = 0.1

    print("Previous model metrics: ", previous_model_properties)
    print("Accuracy: ", accuracy)

    metrics.log_metric("Accuracy", float(accuracy))
    metrics.log_metric("Prev Model Accuracy", float(previous_model_properties["accuracy"]))


@component(base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
           packages_to_install=["pandas==2.2.3",
                                "onnxruntime==1.20.1",
                                "dask[dataframe]==2024.8.0",
                                "scikit-learn==1.6.1"])
def validate_onnx_model(
    version: str,
    onnx_model: Input[Model],
    training_data: Input[Dataset],
    hyperparameters: Input[Artifact],
    torch_metrics: Input[Artifact],
    onnx_metrics: Output[Metrics]
):
    import onnxruntime as rt
    import pandas as pd

    # load parameters
    import json
    parms = {}
    with open(hyperparameters.path, "r") as f:
        parms = json.load(f)

    t_metrics = {}
    with open(torch_metrics.path, "r") as f:
        t_metrics = json.load(f)

    # load tesing dataset
    import torch
    from torch.utils.data import Dataset
    import torch.nn.functional as F
    import numpy as np

    # sklearn
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    # torch custom dataset for this usecase
    class JukeboxDataset(Dataset):
        def __init__(self, dataset_in, features_columns, label_column=None):
            self.features_columns = features_columns
            self.label_column = label_column

            # read datasets and get relevant features inside
            self.dataset_in = pd.read_parquet(dataset_in)
            self.merged_dataset = self.dataset_in[self.features_columns]

            # prepare labels if needed
            if self.label_column:
                self.labels_data = self.dataset_in[self.label_column]
                # encode labels
                self.label_encoder = LabelEncoder()
                encoded_labels = torch.from_numpy(self.label_encoder.fit_transform(self.labels_data))
                self.labels = F.one_hot(encoded_labels)

            # normalize dataset between [-1,1]
            # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier 
            # for the model to learn than random (and potentially large) values.
            # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global 
            # distribution of variables (which is influenced by the test set) into the training set.
            self.minmaxscaler = MinMaxScaler((-1, 1))
            self.merged_dataset = pd.DataFrame(self.minmaxscaler.fit_transform(self.merged_dataset), index=self.merged_dataset.index, columns=self.merged_dataset.columns)

        def __len__(self):
            return len(self.merged_dataset)

        def __getitem__(self, idx):
            """
            Retrieves a single row from the dataset file.
            """
            dp = self.merged_dataset.iloc[idx]

            if self.label_column:
                label = self.labels[idx]
                return torch.tensor(dp.values).float(), label.float()
            else:
                return torch.tensor(dp.values).float()  # For unsupervised learning

    # load testing dataset
    dataset = JukeboxDataset(training_data.path,
                             parms.get("dataset_features"),
                             parms.get("label_feature"))
    test_data = torch.utils.data.Subset(dataset, parms.get("test_indices"))

    # load onnx model
    onnx_inference_model = rt.InferenceSession(onnx_model.path, providers=rt.get_available_providers())

    # get inputs and outputs of the onnx model
    onnx_input = onnx_inference_model.get_inputs()[0]
    onnx_output = onnx_inference_model.get_outputs()[0]
    input_name = onnx_input.name
    output_name = onnx_output.name

    print(f"ONNX Model:\n Input: {input_name}, shape: {onnx_input.shape}\n Output: {output_name}, shape: {onnx_output.shape}")

    # make a prediction using the model
    y_pred_temp = []
    y_test_temp = []
    for point, label in test_data:
        point = point.reshape(onnx_input.shape)
        y_pred_temp.append(onnx_inference_model.run([output_name], {input_name: point.numpy()}))
        label = label.reshape(onnx_output.shape)
        y_test_temp.append(label.numpy())

    # compute prediction max values
    y_pred_argmax = [np.argmax(k[0]) for k in y_pred_temp]

    # compute label max
    y_test_argmax = [np.argmax(k) for k in y_test_temp]

    # compute accuracy
    onnx_accuracy = np.sum([x == y for x,y in zip(y_pred_argmax, y_test_argmax)]) / len(y_pred_argmax)
    print(f"ONNX Model Accuracy: {onnx_accuracy:.4f}")

    # save metrics
    onnx_metrics.log_metric("Torch Model Accuracy", float(t_metrics.get("accuracy")))
    onnx_metrics.log_metric("ONNX Model Accuracy", float(onnx_accuracy))