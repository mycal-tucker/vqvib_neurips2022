import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, comm_dim, recons_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.comm_dim = comm_dim
        self.recons_dim = recons_dim
        self.hidden_dim = 64

        in_dim = comm_dim
        out_dim = self.hidden_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim
        self.final_layer = nn.Linear(self.hidden_dim, recons_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        return self.final_layer(x)
