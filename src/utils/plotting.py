import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
from sklearn.decomposition import PCA

import src.settings as settings


def nat_to_bit(n):
    return n / np.log(2)


def plot_pca(all_data, coloring_data=None, legend=None):
    markers = ["o", "x", "s", "D", "*"][:len(all_data)]
    sizes = [20, 50]
    fig, ax = plt.subplots()
    for data_idx, data in enumerate(all_data):
        if settings.pca is None:
            settings.pca = PCA(n_components=2)
            settings.pca.fit(data)
        if data.shape[1] > 2:
            transformed = settings.pca.transform(data)
            # To actually see all the points around a cluster, we need to add noise to spread them out a tiny bit.
            transformed = np.random.normal(transformed, 0.1)
        else:
            transformed = data
        x = transformed[:, 0]
        y = transformed[:, 1]
        # Pull out coloring info
        coloring = None if coloring_data is None else coloring_data[data_idx]
        pcm = ax.scatter(x, y, s=sizes[data_idx], marker=markers[data_idx], c=coloring[:len(x)])
        if legend is not None:
            handles, labels = pcm.legend_elements(prop='colors')
            ax.legend(handles, legend)


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

    # Plot comm accuracy vs. complexity (all in bits)
    caps = [nat_to_bit(c) for c in metrics.capacities]
    # Communication accuracy is already in bits.
    scatter = plt.scatter(caps, metrics.comm_accs, c=metrics.weights)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Informativeness (bits)")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    if ib_model is not None:
        plt.plot(ib_model.IB_curve[0], ib_model.IB_curve[1], color='black')
    plt.savefig(base_path + 'cap_acc_bits.png')
    plt.close()

    # Plot gNID vs. complexity (in bits)
    caps = [nat_to_bit(c) for c in metrics.capacities]
    scatter = plt.scatter(caps, metrics.gnids, c=metrics.weights)
    plt.xlim([0, 5.0])
    plt.ylim([0, 1.0])
    plt.xlabel("Complexity (bits)")
    plt.ylabel("gNID")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    plt.savefig(base_path + 'gnids.png')
    plt.close()

    # Number of clusters vs. Complexity
    scatter = plt.scatter(caps, metrics.clusters, c=metrics.weights)
    plt.xlabel("Complexity (bits)")
    plt.ylabel("Num clusters")
    cbar = plt.colorbar(scatter, format=tick.FormatStrFormatter('%.2f'))
    cbar.set_label('KL Loss Weight', rotation=270, labelpad=15)
    plt.savefig(base_path + 'cap_clusters.png')
    plt.close()


def plot_modemap(prob_array, ib_model, title, filename):
    plt.figure(figsize=(19.3, 7.5))
    ib_model.mode_map(prob_array)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print('basepath', filename)
    epoch_num = filename.split('/')[-2]
    print("epoch number", epoch_num)
    new_root = '/'.join(filename.split('/')[:-2]) + '/'
    print('new root', new_root)
    plt.savefig(new_root + 'modemaps/' + epoch_num + '.png')
    plt.close()


def plot_color_comms(cielab, color_to_comm, color_to_rgb, base_path):
    cielab = np.vstack(cielab)
    comms = np.vstack(color_to_comm)
    rgbs = np.vstack(color_to_rgb)
    # Come up with some sorting order. Below are two options.
    sorting = np.argsort(cielab, axis=0)[:, 1]  # Which of L*A*B* you want to use.
    # sorting = np.argsort(rgbs, axis=0)[:, 0, 1]  # By greenness
    comms = comms[sorting]
    rgbs = rgbs[sorting]
    plot_pca([comms], coloring_data=[rgbs])
    plt.savefig(base_path + 'pca.png')
    epoch_num = base_path.split('/')[-2]
    plt.savefig(base_path + '../pca/' + epoch_num + '.png')
    plt.close()

    # And a comms heatmap to show how close vectors for colors are to each other.
    comm_dists = np.zeros((len(comms), len(comms)))
    color_dists = np.zeros((len(comms), len(comms)))
    np.set_printoptions(precision=2)
    for i, c1 in enumerate(comms):
        for j, c2 in enumerate(comms):
            comm_dists[i, j] = np.linalg.norm(c1 - c2)
            color_dists[i, j] = np.linalg.norm(rgbs[i] - rgbs[j])
    plt.imshow(comm_dists[:10, :10], cmap='hot', interpolation='nearest')
    plt.savefig(base_path + 'small_heatmap.png')
    plt.imshow(comm_dists, cmap='hot', interpolation='nearest')
    plt.savefig(base_path + 'heatmap.png')
    plt.close()
