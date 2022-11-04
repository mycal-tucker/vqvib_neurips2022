import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Listener, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        in_dim = input_dim
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        y = self.fc2(h)
        return y
