import torch
import torch.nn as nn


# Simple loss function, I can make it more advanced later
def loss(outputs: torch.Tensor, classes: torch.Tensor):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, classes)
    return loss
