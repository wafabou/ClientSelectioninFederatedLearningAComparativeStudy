
import numpy as np
import torch
import torch.utils.data as dataloader
from plato.trainers import basic
from plato.utils import csv_processor
from torch import nn
import os
from plato.config import Config
from plato.utils import csv_processor
import time
import logging
from termcolor import colored
from plato.trainers import basic, loss_criterion
import os


class Trainer(basic.Trainer):
    def process_loss(self, outputs, labels):
            loss_func = loss_criterion.get()
            per_batch_loss = loss_func(outputs, labels)
            self.run_history.update_metric(
                "train_batch_loss", per_batch_loss.cpu().detach().numpy()
            )
            return per_batch_loss

    def get_loss_criterion(self):
            return self.process_loss
