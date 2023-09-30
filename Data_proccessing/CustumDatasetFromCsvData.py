import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class CustomDatasetFromCsvData(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data.label = self.data.label.replace({3: 0, 2: 1, 1: 0})
        self.labels = np.asarray(self.data.iloc[:, 78])
        self.data = self.data.drop(["Unnamed: 0", "subject", "label"], axis=1)
        self.transform = transform

    def __getitem__(self, index):
        y = self.labels[index]
        img = np.asarray(self.data.iloc[index][1:]).reshape(1, 76).astype("uint8")
        return torch.Tensor(img), y

    def __len__(self):
        return len(self.data.index)
