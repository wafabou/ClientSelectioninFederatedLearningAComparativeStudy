#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
import torch.optim as optim
input_dim=76
output_dim = 15
hidden_dim = 10
layer_dim = 3
batch_size = 5
dropout_prob = 0.0
###RNN models
class RNN(nn.Module):
   def __init__(self):
       super(RNN, self).__init__()
        # Defining the number of layers and the nodes in each layer
       self.hidden_dim = hidden_dim
       self.layer_dim = layer_dim
       self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        # Fully connected layer
       self.fc = nn.Linear(hidden_dim, output_dim)

   def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the models
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out




