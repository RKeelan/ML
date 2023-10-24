import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
