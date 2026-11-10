import torch as torch
from abc import abstractmethod, ABC
from dataclasses import dataclass
from gp.tinyverse import Hyperparameters, Config

@dataclass()
class NNConfig(Config):
    report_interval: int
    loss: callable
    ideal: float
    num_inputs: int
    num_epochs: int
    num_outputs: int

@dataclass
class NNHyperparameters(Hyperparameters):
    num_units: int
    num_layers: int
    learning_rate: float
    batch_size: float
    dropout: float
    activation: callable

class NNModel(ABC, torch.nn.Module):
    hyperparameters: NNHyperparameters
    config: NNConfig

    def __init__(self, config_, hyperparameters_):
        super().__init__()
        self.config = config_
        self.hyperparameters = hyperparameters_

    @abstractmethod
    def predict(self, data):
        pass

class TinyTrain:
    config: NNConfig
    hyperparameters: NNHyperparameters
    loss: object

    def __init__(self, config_: NNConfig, hyperparameters_: NNHyperparameters):
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.loss = self.config.loss

    def train(self, model, optimiser, X_train, X_test,  y_train, y_test):

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )

        test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.hyperparameters.batch_size, shuffle=True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hyperparameters.batch_size, shuffle=False
        )

        n_train = len(X_train)
        n_test = len(X_test)

        for epoch in range(self.config.num_epochs):
            running_train_loss = 0.0
            for step, (inputs, targets) in enumerate(train_dataloader):
                output = model(inputs)
                train_loss = self.loss(output, targets)
                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()
                running_train_loss += train_loss.item()

            if epoch % self.config.report_interval == 0:
                predictions = []
                with torch.no_grad():
                    running_test_loss = 0.0
                    for idx, (inputs, targets) in enumerate(test_dataloader):
                        output = model(inputs)
                        predictions.append(output)
                        test_loss = self.loss(output, targets)
                        running_test_loss += test_loss.item()
                train_loss = running_train_loss / n_train
                test_loss = running_test_loss / n_test
                print(f"epoch: {epoch} train_loss: {train_loss} test_loss: {test_loss}")
                if test_loss <= self.config.ideal:
                    print("Ideal test loss has been reached")
                    break
