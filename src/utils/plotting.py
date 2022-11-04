import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
from sklearn.decomposition import PCA

import src.settings as settings
from src.utils.helper_fns import get_complexity, nat_to_bit
from ib_color_naming.src.figures import WCS_CHIPS
from ib_color_naming.src.tools import lab2rgb


def plot_pca(comms, coloring_data, legend=None, sizes=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    if settings.pca is None:
        settings.pca = PCA(n_components=2)
        settings.pca.fit(comms)
    if comms.shape[1] > 2:
        transformed = settings.pca.transform(comms)
    else:
        transformed = comms
    s = 20 if sizes is None else sizes
    pcm = ax.scatter(transformed[:, 0], transformed[:, 1], s=s, marker='o', c=coloring_data)
    if legend is not None:
        handles, labels = pcm.legend_elements(prop='colors')
        ax.legend(handles, legend)
    if settings.ax_limits is None:
        settings.ax_limits = ax.get_xlim() + ax.get_ylim()
    ax.set_xlim(settings.ax_limits[0] - 1, settings.ax_limits[1] + 1)
    ax.set_ylim(settings.ax_limits[2] - 1, settings.ax_limits[3] + 1)
    ax.title.set_text("a. Communication Vectors (2D PCA)")


def plot_training_curve(metrics, base_path, ib_model=None):
    # Plot accuracy vs. Complexity
    caps = [nat_to_bit(c) for c in metrics.capacities]
    scatter = plt.scatter(caps, metrics.accuracies, c=metrics.weights)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Accuracy (percent)")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    plt.savefig(base_path + 'cap_acc_perc.png')
    plt.close()

    # Plot recons loss vs. complexity
    caps = [nat_to_bit(c) for c in metrics.capacities]
    scatter = plt.scatter(caps, metrics.recons, c=metrics.weights)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Recons loss")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    plt.savefig(base_path + 'cap_recons_loss.png')
    plt.close()

    # Plot recons loss vs. incr
    scatter = plt.plot(metrics.recons)
    plt.xlabel("Epoch Idx")
    plt.ylabel("Recons loss")
    plt.savefig(base_path + 'epoch_recons_loss.png')
    plt.close()

    # Plot informativness vs. complexity (all in bits)
    caps = [nat_to_bit(c) for c in metrics.capacities]
    # Communication accuracy is already in bits.
    scatter = plt.scatter(caps, metrics.informativeness, c=metrics.weights)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Informativeness (bits)")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    if ib_model is not None:
        plt.plot(ib_model.IB_curve[0], ib_model.IB_curve[1], color='black')
    plt.savefig(base_path + 'cap_acc_bits.png')
    plt.close()


def plot_many_runs(metric_objects, path, labels=None, ib_model=None, human_data=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    human_comps = []
    human_infos = []
    for lang in human_data.all_langs():
        lang_probs = human_data.lang_pW_C(lang)
        comp = nat_to_bit(get_complexity(lang_probs, ib_model))
        informativeness = ib_model.accuracy(lang_probs)  # Already in bits
        human_comps.append(comp)
        human_infos.append(informativeness)
    plt.scatter(human_comps, human_infos, label='Human Languages')
    # Plot Informativeness vs. complexity
    for i, metrics in enumerate(metric_objects):
        caps = [nat_to_bit(c) for c in metrics.capacities]
        label = metrics.label if labels is None else labels[i]
        plt.scatter(caps, metrics.informativeness, label=label)
    if ib_model is not None:
        plt.plot(ib_model.IB_curve[0], ib_model.IB_curve[1], color='black')
    plt.xlabel("Complexity (bits)", fontsize=20)
    plt.ylabel("Informativeness (bits)", fontsize=20)
    plt.legend(loc='lower right')
    plt.savefig('info', transparent=True, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    # Plot utility vs. complexity
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i, metrics in enumerate(metric_objects):
        label = metrics.label if labels is None else labels[i]
        caps = [nat_to_bit(c) for c in metrics.capacities]
        scatter = plt.scatter(caps, metrics.accuracies, label=label)
    plt.ylabel("Utility %")
    plt.xlabel("Complexity (bits)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('utility', transparent=True)
    plt.close()


def plot_modemap(prob_array, ib_model, title, filename=None):
    ib_model.mode_map(prob_array)
    plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)


def plot_comms_pca(pw_m, speaker, save_path=None, ax=None):
    n = pw_m.shape[0]
    pM = np.ones((n, 1)) / n
    qMW = pw_m * pM
    pW = qMW.sum(axis=0)[:, None]
    pC_W = qMW.T / (pW + 1e-20)
    avg_color_per_comm_id = lab2rgb(pC_W.dot(WCS_CHIPS))
    # Now just look up the actual comms associated with each word
    comm_list = []
    coloring_list = []
    sizes = []
    for comm_id in range(speaker.num_tokens):
        comm_prob = pW[comm_id]
        if comm_prob < 0.0001:
            continue  # Skip comm id that almost never shows up.
        avg_color = avg_color_per_comm_id[comm_id]
        vec = speaker.vq_layer.prototypes[comm_id].detach().cpu().numpy()
        for _ in range(int(comm_prob * 100)):  # Repeat the vectors proportional to likelihood.
            comm_list.append(vec)
            coloring_list.append(avg_color)
            sizes.append(2000 * comm_prob)  # And rescale plotted points proportional to likelihood.
    comm_list = np.vstack(comm_list)
    coloring_list = np.vstack(coloring_list)
    plot_pca(comm_list, coloring_list, sizes=sizes, ax=ax)
    plt.savefig(save_path + '/pca.pdf')
