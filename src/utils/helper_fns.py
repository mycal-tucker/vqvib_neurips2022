import numpy as np
from src.data.wcs_data import WCSDataset
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import src.settings as settings


def get_complexity(w_m, ib_model):
    marginal = np.average(w_m, axis=0, weights=ib_model.pM[:, 0])
    complexities = []
    for likelihood in w_m:
        summed = 0
        for l, p in zip(likelihood, marginal):
            if l == 0:
                continue
            summed += l * (np.log(l) - np.log(p))
        complexities.append(summed)
    return np.average(complexities, weights=ib_model.pM[:, 0])  # Note that this is in nats (not bits)


def nat_to_bit(n):
    return n / np.log(2)


def get_data(ibm, bs):
    raw_data = WCSDataset()
    speaker_obs = []
    listener_obs = []
    labels = []
    probs = []
    for c1 in range(1, 331):
        features1 = raw_data.get_features(c1)
        for c2 in range(1, 331):
            features2 = raw_data.get_features(c2)
            s_obs = features1  # Speaker only sees the target
            if np.random.random() < 0.5:
                l_obs = np.hstack([features1, features2])
                label = 0
            else:
                l_obs = np.hstack([features2, features1])
                label = 1
            speaker_obs.append(s_obs)
            listener_obs.append(l_obs)
            labels.append(label)
            # Also track the prior probability of the color to specify the sampling.
            probs.append(ibm.pM[c1 - 1, 0])

    dataset = TensorDataset(torch.Tensor(np.array(speaker_obs)).to(settings.device),
                            torch.Tensor(np.array(listener_obs)).to(settings.device),
                            torch.Tensor(np.array(labels)).long().to(settings.device))
    sampler = WeightedRandomSampler(probs, num_samples=330 * 330)
    dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler)
    return raw_data, dataloader
