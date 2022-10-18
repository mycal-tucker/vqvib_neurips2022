import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.settings as settings
from src.models.network_utils import reparameterize


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, alpha=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)

    def forward(self, latents):
        dists_to_protos = torch.sum(latents ** 2, dim=1, keepdim=True) + \
                          torch.sum(self.prototypes ** 2, dim=1) - 2 * \
                          torch.matmul(latents, self.prototypes.t())

        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(closest_protos.size(0), self.num_protos).to(settings.device)
        encoding_one_hot.scatter_(1, closest_protos, 1)
        quantized_latents = torch.matmul(encoding_one_hot, self.prototypes)

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.alpha + embedding_loss

        # Approximate the entropy of the distribution for which prototypes are used.
        # Here, we just multiply the approximated entropy by 0.05 and add to the loss, but one can vary this weight
        # to induce more or fewer distinct VQ clusters.
        ent = self.get_categorical_ent(dists_to_protos)
        vq_loss += settings.entropy_weight * ent

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss

    def get_categorical_ent(self, distances):
        # Approximate the onehot of which prototype via a softmax of the negative distances
        logdist = torch.log_softmax(-distances, dim=1)
        soft_dist = torch.mean(logdist.exp(), dim=0)
        epsilon = 0.000001  # Need a fuzz factor for numerical stability.
        soft_dist += epsilon
        soft_dist = soft_dist / torch.sum(soft_dist)
        logdist = soft_dist.log()
        entropy = torch.sum(-1 * soft_dist * logdist)
        return entropy


class VQVIB(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_protos):
        super(VQVIB, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.num_tokens = num_protos  # Need this general variable for num tokens
        in_dim = input_dim
        out_dim = self.hidden_dim if num_layers > 1 else output_dim
        self.layers = nn.ModuleList()
        while len(self.layers) < num_layers:
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            out_dim = self.hidden_dim if len(self.layers) < num_layers - 1 else output_dim
        self.vq_layer = VQLayer(num_protos, output_dim)
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_var = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = F.relu(x)
        logvar = self.fc_var(x)
        mu = self.fc_mu(x)
        sample = reparameterize(mu, logvar)
        # Quantize the vectors
        output, quantization_loss = self.vq_layer(sample)
        # Compute the KL divergence
        divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        # Total loss is the penalty on complexity plus the quantization (and entropy) losses.
        total_loss = settings.kl_weight * divergence + quantization_loss
        return output, total_loss, divergence

    # Helper method calculates the distribution over prototypes given an input. It largely recreates the forward pass
    # but is slightly faster because it does not compute categorical entropy.
    def get_token_dist(self, x):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = F.relu(x)
            logvar = self.fc_var(x)
            mu = self.fc_mu(x)
            samples = reparameterize(mu, logvar)
            # Now discretize.
            dists_to_protos = torch.sum(samples ** 2, dim=1, keepdim=True) + \
                              torch.sum(self.vq_layer.prototypes ** 2, dim=1) - 2 * \
                              torch.matmul(samples, self.vq_layer.prototypes.t())
            closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
            encoding_one_hot = torch.zeros(closest_protos.size(0), self.vq_layer.num_protos).to(settings.device)
            encoding_one_hot.scatter_(1, closest_protos, 1)
            likelihoods = np.mean(encoding_one_hot.detach().cpu().numpy(), axis=0)
        return likelihoods
