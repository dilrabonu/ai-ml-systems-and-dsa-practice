import torch
import torch.nn as nn

model = Sequential(
    nn.Linear(20,64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32,1)
)