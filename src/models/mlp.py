import torch.nn as nn
import torch.nn.functional as F
from src.models.network_utils import reparameterize, gumbel_softmax
import torch
import src.settings as settings
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, onehot):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.onehot = onehot
        self.num_tokens = output_dim
        in_dim = input_dim
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else output_dim
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_var = nn.Linear(output_dim, output_dim)
        self.disc_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        if not self.onehot:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            output = reparameterize(mu, logvar)
            # For a fixed, unit gaussian
            capacity = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        else:
            output, dist = gumbel_softmax(x, hard=True, return_dist=True)
            logits = dist.log()  # Needs the log of the input
            prior = torch.ones_like(logits) / logits.shape[1]  # Assume uniform prior
            # The order of the arguments matters a lot.
            capacity = self.disc_loss_fn(logits, prior)
        network_loss = settings.kl_weight * capacity
        return output, network_loss, capacity

    def get_token_dist(self, x):
        assert self.onehot, "No categorical distribution if not onehot"
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i != len(self.layers) - 1:
                    x = F.relu(x)
            _, dist = gumbel_softmax(x, hard=True, return_dist=True)
            likelihoods = np.mean(dist.detach().cpu().numpy(), axis=0)
        return likelihoods
