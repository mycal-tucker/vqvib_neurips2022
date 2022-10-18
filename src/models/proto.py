import torch.nn as nn
import torch
import torch.nn.functional as F
from src.models.network_utils import gumbel_softmax, reparameterize
import src.settings as settings


class ProtoNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos):
        super(ProtoNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.num_tokens = num_protos
        in_dim = input_dim
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else num_protos
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, output_dim))
        self.prototypes.data.uniform_(-1 / num_protos, 1 / num_protos)
        self.fc_var = nn.Linear(num_protos, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        # Turn into onehot to multiply by protos.
        onehot_pred = gumbel_softmax(x, hard=True)
        proto = torch.matmul(onehot_pred, self.prototypes)
        logvar = self.fc_var(x)
        # Sample around the prototype
        output = reparameterize(proto, logvar)
        # Compute the KL divergence
        divergence = torch.mean(-0.5 * torch.sum(1 + logvar - proto ** 2 - logvar.exp(), dim=1), dim=0)
        total_loss = settings.kl_weight * divergence
        return output, total_loss, divergence
