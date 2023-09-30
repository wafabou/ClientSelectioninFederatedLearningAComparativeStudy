import logging
import os
import time

import numpy as np
import torch
import torch.utils.data as dataloader
from plato.config import Config
from plato.trainers import basic
from plato.utils import csv_processor
from sklearn.metrics import confusion_matrix
from termcolor import colored
from torch import nn
class Trainer(basic.Trainer):
    def process_loss(self, outputs, labels) -> torch.Tensor:
        loss_func = nn.CrossEntropyLoss(reduction="none")
        per_sample_loss = loss_func(outputs, labels)
        self.run_history.update_metric(
            "train_squared_loss_step",
            sum(np.power(per_sample_loss.cpu().detach().numpy(), 2)),
        )
        return torch.mean(per_sample_loss)
    def get_loss_criterion(self):
        return self.process_loss
