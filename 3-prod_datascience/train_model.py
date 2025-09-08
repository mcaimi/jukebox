# Import objects from KubeFlow Pipelines DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
)


@component(base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
           packages_to_install=["pandas==2.2.3",
                                "dask[dataframe]==2024.8.0",
                                "scikit-learn==1.6.1"])
def train_torch_model(
    version: str,
    train_dataset: Input[Dataset],
    hyperparameters: dict,
    trained_model: Output[Model],
    minmax_scaler_model: Output[Model],
    label_encoder_model: Output[Model],
    training_parameters: Output[Artifact]
):
    """
    Trains a PyTorch NN Model.
    """

    # import relevant libraries for the dataset preparation
    import os
    import random

    # data libraries
    import pandas as pd
    import numpy as np

    # sklearn
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    # pytorch
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    import torch.nn.functional as F
    import torch.cuda as tc
    from torch.optim import Adam

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

    # Set random seeds
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    """ Load Dataset into Memory and Preprocess it """

    # features...
    label_feature = 'country'
    dataset_features = ['is_explicit',
                        'duration_ms',
                        'danceability',
                        'energy',
                        'key',
                        'loudness',
                        'mode',
                        'speechiness',
                        'acousticness',
                        'instrumentalness',
                        'liveness',
                        'valence',
                        'tempo']

    # load datasets
    song_dataset = JukeboxDataset(dataset_in=train_dataset.path,
                                  features_columns=dataset_features,
                                  label_column=label_feature)

    # print dataset info
    print(f"Dataset Contains {len(song_dataset)} items.")

    """ Prepare train/validate splits """

    # Split the data into training and testing sets so you have something to test the trained model with.
    train_percentage = 0.8
    training_data, test_data = random_split(song_dataset, [train_percentage, 1.0-train_percentage])

    """ Training Hyperparameters """
    # get or declare parameters
    n_samples, n_feats = song_dataset.merged_dataset.shape
    n_rows, n_classes = song_dataset.labels.shape
    base_size = hyperparameters.get("base_size")
    learning_rate = hyperparameters.get("learning_rate")
    batch_size = hyperparameters.get("batch_size")
    epochs = hyperparameters.get("epochs")

    # prepare output parameters artifact
    parm_dict = {
        "n_samples": n_samples,
        "n_feats": n_feats,
        "n_rows": n_rows,
        "n_classes": n_classes,
        "base_size": base_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "training_indices": training_data.indices,
        "test_indices": test_data.indices,
        "label_feature": label_feature,
        "dataset_features": dataset_features,
    }

    # detect device
    device = "cpu"
    if tc.is_available():
        device = "cuda"

    print(f"DEVICE:\n Training on {device}")

    # Import torch specific libraries
    from torch.nn import Module, Sequential, ReLU, Linear, CrossEntropyLoss
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

    # instantiate a model object
    model = CountryPredictorNetwork(n_feats, base_size, n_classes)

    print(f"Model structure: {model}\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} {param.dtype}\n")

    """ Run Training! """
    # prepare loss functions and optimizers
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # model training function
    def training_loop(model, dataloader, loss_function, optimizer, epoch, device="cpu"):
        b_size = len(dataloader.dataset)
        model.to(device)
        model.train()
        batch_loss = 0.0
        for b, (point, label) in enumerate(dataloader):
            # compute prediction
            pred = model(point.to(device))
            # measure loss
            loss = loss_function(pred.to(device), label.to(device))

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate loss
            batch_loss += loss.item()

            if b % 1000 == 0:
                current_loss, batch_position = loss.item(), b * batch_size
                print(f" -> Running Loss: {current_loss:.4f}, Processed Samples: [{batch_position}/{b_size}]")

        # epoch update
        print(f"EPOCH {epoch}: Cumulative Loss: {batch_loss/b_size:.4f}")

    # data loaders
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # run the training loop for some epochs
    for i in range(epochs):
        print(f"============ EPOCH {i}/{epochs} ============")
        training_loop(model, training_dataloader, loss_fn, optimizer, i, device)

    print("Training of model is complete")

    # save model to disk
    trained_model._set_path(trained_model.path + ".pt")
    torch.save(model.state_dict(), trained_model.path)

    # save scaler and label encoder
    import pickle
    minmax_scaler_model._set_path(minmax_scaler_model.path + ".pkl")
    with open(minmax_scaler_model.path, "wb") as handle:
        pickle.dump(song_dataset.minmaxscaler, handle)
    label_encoder_model._set_path(label_encoder_model.path + ".pkl")
    with open(label_encoder_model.path, "wb") as handle:
        pickle.dump(song_dataset.label_encoder, handle)

    # save training parameters to be used in following steps
    import json
    training_parameters._set_path(training_parameters.path + ".json")
    with open(training_parameters.path, "w") as tp:
        json.dump(parm_dict, tp)