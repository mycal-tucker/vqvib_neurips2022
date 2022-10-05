import numpy as np
from src.models.team import Team
from src.models.mlp import MLP
from src.models.vq import VQ
from src.models.decoder import Decoder
from src.models.proto import ProtoNetwork
from src.data.wcs_data import WCSDataset
from src.utils.plotting import plot_training_curve, plot_color_comms, plot_modemap
import torch
import torch.nn as nn
from src.utils.performance_metrics import PerformanceMetrics
from ib_color_naming.src import ib_naming_model
from ib_color_naming.src.tools import gNID
from skimage.color import lab2rgb
import src.settings as settings
import pickle

import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


def get_data():
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
            probs.append(ib_model.pM[c1 - 1, 0])

    dataset = TensorDataset(torch.Tensor(np.array(speaker_obs)).to(settings.device),
                            torch.Tensor(np.array(listener_obs)).to(settings.device),
                            torch.Tensor(np.array(labels)).long().to(settings.device))
    sampler = WeightedRandomSampler(probs, num_samples=330 * 330)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return raw_data, dataloader


def train(model, raw_data, data, save_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    settings.kl_weight = 0.05  # Initialize the weight as non-zero just to prevent total collapse.
    capacities = []
    accs = []
    recons_losses = []
    comm_accs = []
    gnids = []
    weights = []
    clusters = []
    if not os.path.exists(save_dir + '/pca'):
        os.makedirs(save_dir + '/pca')
        os.makedirs(save_dir + '/modemaps')
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        base_path = save_dir + str(epoch) + '/'
        if epoch > burnin_epochs:
            settings.kl_weight += kl_incr
        print("KL loss weight", settings.kl_weight)
        running_loss = 0.0
        running_recons_loss = 0.0
        running_sl = 0.0
        running_info = 0.0
        num_correct = 0
        num_total = 0
        num_batches = 0
        for i, batch in enumerate(data):
            num_batches += 1
            true_color, listener_obs, labels = batch
            noisy_obs = np.random.normal(true_color.cpu(), np.sqrt(obs_noise_var))
            speaker_obs = torch.Tensor(noisy_obs).to(settings.device)
            optimizer.zero_grad()

            outputs, speaker_loss, info, recons = model(speaker_obs, listener_obs)
            # Signal from the listener predicting which was the target
            loss = criterion(outputs, labels)
            # Reconstruction loss is KL divergence between input and recons dist. Because there's a fixed input
            # variance, we ignore that term
            r_mu, _ = recons
            recons_loss = torch.mean(0.5 * torch.sum(((true_color - r_mu) ** 2) / obs_noise_var, dim=1), dim=0)
            loss += recons_weight * recons_loss

            # Metrics
            pred_labels = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            num_correct += np.sum(pred_labels == labels.cpu().numpy())
            num_total += pred_labels.size
            loss += speaker_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_sl += speaker_loss.item()
            running_recons_loss += recons_loss.item()
            running_info += info.item()
        if epoch < burnin_epochs - 1:
            continue
        kl_loss = running_info / num_batches
        acc = num_correct / num_total
        recons_loss_val = running_recons_loss / num_total
        print("Accuracy", acc)
        comm_acc = None
        gnid = None
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        capacities.append(kl_loss)
        accs.append(acc)
        recons_losses.append(recons_loss_val)
        weights.append(settings.kl_weight)
        if ((isinstance(model.speaker, VQ)) or
                                   (isinstance(model.speaker, MLP) and model.speaker.onehot)):
            complexity, comm_acc, gnid = gen_pw_u(model.speaker, raw_data, base_path, do_plot=epoch % 5 == 3)
            # We get the actual complexity, so overwrite the KL proxy.
            print("Complexity in bits", complexity / np.log(2))
            print("Comm acc", comm_acc)
            capacities[-1] = complexity
            print("GNID", gnid)
        comm_accs.append(comm_acc)
        gnids.append(gnid)
        num_clusters = None
        if epoch % plotting_freq == plotting_freq - 1:
            num_clusters = plot_comms(model.speaker, raw_data, base_path)
        clusters.append(num_clusters)
        print("Num Clusters", num_clusters)
        plot_training_curve(PerformanceMetrics(accs, capacities, recons_losses, comm_accs, gnids, weights, clusters, None), base_path, ib_model=ib_model)
    print("Accuracies", accs)
    print("Comm accuracies", comm_accs)
    print("Capacities", capacities)
    print("Num clusters", clusters)
    return accs, capacities, recons_losses, comm_accs, gnids, weights, clusters


def plot_comms(speaker, data, base_path):
    full_cielab = []
    full_color_to_comm = []
    full_color_to_rgb = []
    cielab = []
    color_to_comm = []
    color_to_rgb = []
    speaker.viz_mode = True
    for c1 in range(1, 331):
        target_features = data.get_features(c1)
        cielab.append(target_features)
        features = target_features
        with torch.no_grad():
            comms, _, _ = speaker(torch.Tensor(features).unsqueeze(0).to(settings.device))
            color_to_comm.append(comms.detach().cpu().numpy())
            expanded_color = np.expand_dims(np.expand_dims(target_features, 0), 0)
            rgb = lab2rgb(expanded_color)
            color_to_rgb.append(rgb)
    speaker.viz_mode = False
    coarse_comms = np.vstack(color_to_comm).round(decimals=1)
    num_coarse_unique = len(np.unique(coarse_comms, axis=0))
    plot_color_comms(cielab, color_to_comm, color_to_rgb, base_path)
    full_cielab.append(cielab)
    full_color_to_comm.append(color_to_comm)
    full_color_to_rgb.append(color_to_rgb)
    return num_coarse_unique


def gen_pw_u(speaker, data, base_path, do_plot=False):
    comp, comm_acc, gnid = None, None, None
    w_m = np.zeros((330, speaker.num_tokens))
    bs = 1000  # How many samples for a single color. Higher is more accurate but slower.
    for c1 in range(1, 331):  # 330 colors in the WCS data.
        features = data.get_features(c1)
        inp_len = feature_len
        noisy_obs = np.random.normal(features, np.sqrt(obs_noise_var), size=(bs, inp_len))
        features = torch.Tensor(noisy_obs).to(settings.device)
        with torch.no_grad():
            likelihoods = speaker.get_token_dist(features)
            w_m[c1 - 1] = likelihoods
    # Save the array so it can be used by IB code.
    with open(base_path + 'pw_m', 'wb') as file:
        pickle.dump(w_m, file)
    # Calculate the divergence from the marginal over tokens. Here we just calculate the marginal
    # from observations. It's very important to weight by the prior.
    marginal = np.average(w_m, axis=0, weights=ib_model.pM[:, 0])
    complexities = []
    for likelihood in w_m:
        summed = 0
        for l, p in zip(likelihood, marginal):
            if l == 0:
                continue
            summed += l * (np.log(l) - np.log(p))
        complexities.append(summed)
    comp = np.average(complexities, weights=ib_model.pM[:, 0])  # Note that this is in nats (not bits)
    # And calculate the communication accuracy
    comm_acc = ib_model.accuracy(w_m)  # Already in bits
    # And just plot the mode map directly.
    # These calls to the ib_color_naming code seem to cause a memory leak. Odd.
    # FIXME: Calls to the ib_color_naming code seem to cause a memory leak, which eventually crashes the program.
    if do_plot:
        _, gnid, bl, qw_m_fit = ib_model.fit(w_m)
        # Plot a few modemaps related to the learned system.
        # 1) What is the modemap of the actual system
        plot_modemap(w_m, ib_model, 'Trained IB System', base_path + 'trained_modemap.png')
        # 2) What is the modemap of the optimal system at that complexity.
        # plot_modemap(qw_m_fit, ib_model, 'Optimal IB System', base_path + 'optimal_modemap.png')
        # 3) What is the modemap of the nearest human language (by gNID)
        # plot_closest_lang(w_m, base_path)
    return comp, comm_acc, gnid


def plot_closest_lang(pw_u, base_path):
    lang_gnids = [gNID(pw_u, raw_data.lang_pW_C(lang_id), ib_model.pM) for lang_id in raw_data.all_langs()]
    best_lang = np.argmin(lang_gnids)
    lang_name = raw_data.lang_name(best_lang)
    print("Closest language", lang_name)
    plot_modemap(raw_data.lang_pW_C(best_lang), ib_model, 'Closest Human Lang: ' + lang_name, base_path + 'human_modemap.png')


def run():
    listener = MLP(comm_dim + (num_distractors + 1) * feature_len, (num_distractors + 1), num_layers=2, onehot=False)
    decoder = Decoder(comm_dim, feature_len, num_layers=1)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    suffix = str(seed)
    if isinstance(speaker, MLP):
        prepend = 'onehot' if speaker.onehot else 'cont'
        suffix = prepend + '_' + suffix
    savepath = 'saved_data/' + speaker.__class__.__name__ + '_' + suffix
    incr_dir = savepath + '_incr/'

    accs, capacities, recons_losses, comm_accs, gnids, weights, clusters = train(model, raw_data, data, incr_dir)
    metrics = PerformanceMetrics(accs, capacities, recons_losses, comm_accs, gnids, weights, clusters, speaker.__class__.__name__)
    metrics.to_file(savepath)  # Save the data to a file. You can load these files for more complex plotting if you'd like.


if __name__ == '__main__':
    feature_len = 3             # Colors are represented in CIELAB space
    num_distractors = 1         # How many distractors does the listener agent see?
    num_epochs = 84
    batch_size = 1024
    comm_dim = 32               # Communication dimension. If you train onehot agents, you may want to increase this.
    burnin_epochs = 30          # How many epochs to train before starting to anneal the complexity parameter.
    obs_noise_var = 64          # Variance for Gaussian observation noise
    plotting_freq = 1           # How often to plot different metrics. Technically, plotting slows things down a bit...
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recons_weight = 1.0        # Scalar weight for reconstruction loss. lambda_I in the neurips paper
    settings.pca = None
    ib_model = ib_naming_model.load_model()
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_data, data = get_data()

    speaker_type = 'VQVIB'
    # speaker_type = 'Onehot'
    # speaker_type = 'Proto'
    if speaker_type == 'VQVIB':
        speaker = VQ(feature_len, comm_dim, num_layers=3, num_protos=330)
        kl_incr = 0.01
    elif speaker_type == 'Onehot':
        comm_dim = 330  # The vocabulary size of onehot agents is tied to the comm_dim
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True)
        kl_incr = 0.1
    elif speaker_type == 'Proto':
        speaker = ProtoNetwork(feature_len, comm_dim, num_layers=3, num_protos=330)
        kl_incr = 0.1
    else:
        assert False, "Need a speaker to be part of [VQVIB, Onehot, Proto]"
    run()
