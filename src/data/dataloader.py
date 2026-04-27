import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


class NormalDataloader(Dataset):
    def __init__(self, dataframe):
        self.X = torch.tensor(dataframe.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], idx


def load_data(df, categorical_cols=None, numerical_cols=None, k=None, batch_size=None, generator=None):
    scaler = MinMaxScaler()
    categorical_cols = categorical_cols if categorical_cols is not None else []
    numerical_cols = numerical_cols if numerical_cols is not None else []

    if not isinstance(categorical_cols, list) or not isinstance(numerical_cols, list):
        raise ValueError("Both 'categorical_cols' and 'numerical_cols' must be a 'list' object type.")

    if not categorical_cols and not numerical_cols:
        raise ValueError("You need to define 'categorical_cols' and/or 'numerical_cols'")

    missing_cols = [col for col in (numerical_cols + categorical_cols) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns not found in the dataframe: {missing_cols}")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("Number of clusters ('k = __') must be provided. E.g., how many should there be?")

    if df.isnull().values.any():
        raise ValueError("Data contains missing values. Please handle NaNs before using this function")

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dtype=float)

    if numerical_cols:
        df[numerical_cols] = df[numerical_cols].astype(float)
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    shape = df.shape[1]
    dataset = NormalDataloader(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    return dataset, dataloader, k, shape, df
