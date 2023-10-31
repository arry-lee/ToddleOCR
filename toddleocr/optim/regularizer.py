"""
criterion = torch.nn.CrossEntropyLoss()
l1_decay = L1Decay(weight_decay=0.01)
l2_decay = L2Decay(weight_decay=0.01)
parameters = model.parameters()

loss = criterion(outputs, labels) + l1_decay(parameters) + l2_decay(parameters)
"""
import torch
import torch.nn as nn


class L1Decay(nn.Module):
    def __init__(self, weight_decay=0.01):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, parameters):
        decay = torch.tensor(0.0, dtype=torch.float32)
        for param in parameters:
            decay = decay + torch.norm(param, p=1)
        return self.weight_decay * decay


class L2Decay(nn.Module):
    def __init__(self, weight_decay=0.01):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, parameters):
        decay = torch.tensor(0.0, dtype=torch.float32)
        for param in parameters:
            decay += torch.norm(param, p=2)
        return 0.5 * self.weight_decay * decay * decay
