import torch as torch
import torch.nn as nn
from nn.nn import NNModel


class FeedForward(NNModel):
    layers: nn.ModuleList
    num_units: int
    num_inputs: int
    num_outputs: int

    def __init__(self, config_, hyperparameters_):
        super(FeedForward, self).__init__(config_, hyperparameters_)
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.num_layers = self.hyperparameters.num_layers
        self.num_units = self.hyperparameters.num_units
        self.num_inputs = self.config.num_inputs
        self.num_outputs = self.config.num_outputs

        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.num_inputs, self.num_units))

        for _ in range(1, self.num_layers - 1):
            self.layers.append(torch.nn.Linear(self.num_units, self.num_units))

        self.layers.append(torch.nn.Linear(self.num_units, self.num_outputs))

        self.activation = self.hyperparameters.activation
        self.dropout = nn.Dropout(self.hyperparameters.dropout)

    def predict(self, data):
        return self.forward(data)

    def forward(self, x):
        out = self.layers[0](x)
        for idx in range(1, self.num_layers - 1):
            out = self.activation(self.layers[idx](out))
            out = self.dropout(out)
        out = self.layers[-1](out)
        return out
