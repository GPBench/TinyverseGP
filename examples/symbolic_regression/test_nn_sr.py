import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from src.nn.nn import NNConfig, NNHyperparameters, TinyTrain
from src.nn.tiny_ff import FeedForward
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark, koza1
from sklearn.model_selection import train_test_split
import torch as torch
import torch.nn as nn


hyperparameters = NNHyperparameters (
    num_units = 100,
    num_layers = 5,
    learning_rate = 1e-4,
    batch_size = 1,
    dropout = 0.5,
    activation = torch.sigmoid
)

config = NNConfig (
    report_interval = 10,
    loss = nn.MSELoss(),
    ideal = 1e-2,
    num_inputs = 1,
    num_epochs = 10000,
    num_outputs = 1
)

model = FeedForward(config, hyperparameters)
trainer = TinyTrain(config, hyperparameters)
optimiser = torch.optim.SGD(model.parameters(), lr=hyperparameters.learning_rate)
benchmark = SRBenchmark()
objective = koza1

X, y = benchmark.random_set(min=-1.0, max=1.0, n=50, objective=objective, dim=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

trainer.train(model, optimiser, X_train, X_test, y_train, y_test)