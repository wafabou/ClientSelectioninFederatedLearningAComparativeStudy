import pandas as pd
import torch
from CustumDatasetFromCsvData import CustomDatasetFromCsvData
from plato.config import Config
from plato.datasources import base


class WESAD(base.DataSource):
    def __init__(self):
        super().__init__()
        train_dset = CustomDatasetFromCsvData("data/WESAD/our_train_data_all.csv")
        test_dset = CustomDatasetFromCsvData("data/WESAD/our_test_data_all.csv")
        self.trainset, self.testset = train_dset, test_dset
