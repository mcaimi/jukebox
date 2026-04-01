try:
    # data libraries
    import pandas as pd

    # sklearn
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder

    # pytorch
    import torch
    from torch.utils.data import Dataset
    import torch.nn.functional as F
except ImportError as e:
    print(f"{e}")


# torch custom dataset for this usecase
class JukeboxDataset(Dataset):
    def __init__(self, rankings_file, properties_file, features_columns, label_column=None):
        self.features_columns = features_columns
        self.label_column = label_column

        # read ranking data from parquet file
        self.rankings_file = pd.read_parquet(rankings_file)
        # remove missing values from the dataset (NaN)
        self.rankings_file.dropna()

        # read songs properties from parquet file
        self.properties_file = pd.read_parquet(properties_file)

        # join datasets and get relevant features inside
        self.merged_dataset_full = self.rankings_file.merge(self.properties_file, on='spotify_id', how='left')
        self.merged_dataset = self.merged_dataset_full[self.features_columns]

        # prepare labels if needed
        if self.label_column:
            self.labels_data = self.rankings_file[self.label_column]
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


# load a merged dataset
class MergedDataset(Dataset):
    def __init__(self, merged_dataset, features, label_column=None):
        self.merged_dataset_full = pd.read_parquet(merged_dataset)
        self.label_column = label_column

        # remove label from dataset
        self.features = features
        self.merged_dataset = self.merged_dataset_full[self.features]

        # prepare labels if needed
        if self.label_column:
            self.labels_data = self.merged_dataset_full[label_column]
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