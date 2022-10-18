import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src.settings as settings
from ib_color_naming.src import ib_naming_model
from ib_color_naming.src.tools import gNID
from src.models.decoder import Decoder
from src.models.mlp import MLP
from src.models.proto import ProtoNetwork
from src.models.team import Team
from src.models.vqvib import VQVIB
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.helper_fns import get_complexity, get_data
from src.utils.plotting import plot_training_curve, plot_comms_pca, plot_modemap


def train(model, raw_data, data, save_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    capacities = []
    accs = []
    recons_losses = []
    infos = []
    weights = []
    if not os.path.exists(save_dir + '/pca'):
        os.makedirs(save_dir + '/pca')
        os.makedirs(save_dir + '/modemaps')
    latest_metrics = None
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        base_path = save_dir + str(epoch) + '/'
        if epoch > burnin_epochs:
            settings.kl_weight += kl_incr
            settings.entropy_weight = 0.1
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
            recons_loss = torch.mean(0.5 * torch.sum(((true_color - recons) ** 2) / obs_noise_var, dim=1), dim=0)
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
        informativeness = None
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        capacities.append(kl_loss)
        accs.append(acc)
        recons_losses.append(recons_loss_val)
        weights.append(settings.kl_weight)
        if ((isinstance(model.speaker, VQVIB)) or
                                   (isinstance(model.speaker, MLP) and model.speaker.onehot)):
            complexity, informativeness = eval_comms(model.speaker, raw_data, base_path, epoch)
            # We get the actual complexity, so overwrite the KL proxy.
            print("Complexity in bits", complexity / np.log(2))
            print("Informativeness", informativeness)
            capacities[-1] = complexity
        infos.append(informativeness)
        latest_metrics = PerformanceMetrics(accs, capacities, recons_losses, infos, weights, None)
        latest_metrics.to_file(base_path + 'metrics.pkl')
        plot_training_curve(latest_metrics, base_path, ib_model=ib_model)
    # Save the metrics from the last epoch, which contain data from the whole run. These data can be plotted via
    # plot_results.py and compared to other runs.
    latest_metrics.to_file('saved_data/' + speaker.__class__.__name__ + '_' + str(seed))


def eval_comms(speaker, data, base_path, epoch_number):
    # Inducing a consistent visualization that's useful across epochs is tough. Here, we set when to reset the PCA
    # for visualizing communication, but one can clearly change this condition.
    if epoch_number % 10 == 0:
        settings.pca = None
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
    # Calculate the divergence from the marginal over tokens. Here we just calculate the marginal
    # from observations. It's very important to weight by the prior.
    comp = get_complexity(w_m, ib_model)
    informativeness = ib_model.accuracy(w_m)  # Already in bits
    # These calls to the ib_color_naming code seem to cause a memory leak. Odd.
    # FIXME: Calls to the ib_color_naming code seem to cause a memory leak, which eventually crashes the program.
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 4]})
    fig.suptitle('Emergent communication in color reference game via VQ-VIB ($\lambda_C = $' + "{:.2f}".format(
        settings.kl_weight) + ')')

    axes[0].title.set_text("a. Communication Vectors (2D PCA)")
    # Plot just information about the agent communication
    _, _, bl, qw_m_fit = ib_model.fit(w_m)
    plot_modemap(w_m, ib_model, title="b. Speaker discretization of color chips")
    if isinstance(speaker, VQVIB):
        plot_comms_pca(w_m, speaker, save_path=base_path, ax=axes[0])
    plt.savefig(base_path + '../pca/' + str(epoch_number) + '.png')
    plt.close()
    # Also plot interesting other modemaps if you would like
    # What is the modemap of the optimal system at that complexity.
    # plot_modemap(qw_m_fit, ib_model, title='Optimal IB System', filename=base_path + '/../modemaps/optimal_' + str(epoch_number) + '.png')
    # What is the modemap of the nearest human language (by gNID)
    # plot_closest_lang(w_m, base_path + '/../modemaps/human_' + str(epoch_number) + '.pdf')

    # Measure how often being closer in the color space means being closer in comm space.
    colors = []
    comms = []
    for c1 in range(1, 331):
        features = data.get_features(c1)
        inp_len = feature_len
        noisy_obs = np.random.normal(features, np.sqrt(0), size=(1, inp_len))
        features = torch.Tensor(noisy_obs).to(settings.device)
        with torch.no_grad():
            comm, _, _ = speaker(features)
        comms.append(comm.detach().cpu().numpy())
        colors.append(noisy_obs)
    comm_dists = np.zeros((len(comms), len(comms)))
    color_dists = np.zeros((len(comms), len(comms)))
    for i, c1 in enumerate(comms):
        for j, c2 in enumerate(comms):
            comm_dists[i, j] = np.linalg.norm(c1 - c2)
            color_dists[i, j] = np.linalg.norm(colors[i] - colors[j])
    comm_dists = np.reshape(comm_dists, (-1, 1))
    color_dists = np.reshape(color_dists, (-1, 1))
    num_samples = 1000
    num_agree = 0
    for _ in range(num_samples):
        idx1 = int(np.random.random() * len(comm_dists))
        idx2 = int(np.random.random() * len(comm_dists))
        comm_dist1 = comm_dists[idx1]
        color_dist1 = color_dists[idx1]
        comm_dist2 = comm_dists[idx2]
        color_dist2 = color_dists[idx2]
        if (comm_dist1 < comm_dist2 and color_dist1 < color_dist2) or \
                (comm_dist1 > comm_dist2 and color_dist1 > color_dist2):
            num_agree += 1
    print("Color/comm agreement", num_agree / num_samples)
    return comp, informativeness


def plot_closest_lang(pw_u, savepath):
    all_langs = sorted(raw_data.all_langs())
    lang_gnids = [gNID(pw_u, raw_data.lang_pW_C(lang_id), ib_model.pM) for lang_id in all_langs]
    best_lang = np.argmin(lang_gnids) + 1  # Note that languages are one-indexed in other code.
    lang_name = raw_data.lang_name(best_lang)
    print("Closest language", lang_name)
    plot_modemap(raw_data.lang_pW_C(best_lang), ib_model, 'Closest Human Lang: ' + lang_name, savepath)


def run():
    listener = MLP(comm_dim + (num_distractors + 1) * feature_len, (num_distractors + 1), num_layers=2, onehot=False,
                   deterministic=True)
    decoder = Decoder(comm_dim, feature_len, num_layers=1)
    model = Team(speaker, listener, decoder)
    model.to(settings.device)

    suffix = str(seed)
    if isinstance(speaker, MLP):
        prepend = 'onehot' if speaker.onehot else 'cont'
        suffix = prepend + '_' + suffix
    savepath = 'saved_data/' + speaker.__class__.__name__ + '_' + suffix
    incr_dir = savepath + '_incr/'
    train(model, raw_data, data, incr_dir)


if __name__ == '__main__':
    feature_len = 3             # Colors are represented in CIELAB space
    num_distractors = 1         # How many distractors does the listener agent see?
    num_epochs = 200             # How many epochs to train for.
    batch_size = 1024
    comm_dim = 32               # Communication dimension. If you train onehot agents, gets reset to 330.
    burnin_epochs = 30          # How many epochs to train before starting to anneal the complexity parameter.
    obs_noise_var = 64          # Variance for Gaussian observation noise
    plotting_freq = 1           # How often to plot different metrics. Technically, plotting slows things down a bit...
    settings.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recons_weight = 1.0        # Scalar weight for reconstruction loss. lambda_I in the neurips paper
    settings.ax_limits = None
    settings.pca = None
    settings.entropy_weight = 0.02
    ib_model = ib_naming_model.load_model()
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw_data, data = get_data(ib_model, batch_size)

    settings.kl_weight = 0.05  # Set some low (but non-zero) initial value for penalizing kl_weight.
    speaker_type = 'VQVIB'
    # speaker_type = 'Onehot'
    # speaker_type = 'Proto'
    if speaker_type == 'VQVIB':
        speaker = VQVIB(feature_len, comm_dim, num_layers=3, num_protos=330)
        kl_incr = 0.02
    elif speaker_type == 'Onehot':
        comm_dim = 330  # The vocabulary size of onehot agents is tied to the comm_dim
        speaker = MLP(feature_len, comm_dim, num_layers=3, onehot=True)
        kl_incr = 0.1
        settings.kl_weight = 0.05  # Seems to need a greater value to prevent numerical issues when annealing starts.
    elif speaker_type == 'Proto':
        speaker = ProtoNetwork(feature_len, comm_dim, num_layers=3, num_protos=330)
        kl_incr = 0.1
    else:
        assert False, "Need a speaker to be part of [VQVIB, Onehot, Proto]"
    run()
