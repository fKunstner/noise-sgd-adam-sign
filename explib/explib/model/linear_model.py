import torch


class LinearModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, X):
        out = self.linear(X)
        return out
