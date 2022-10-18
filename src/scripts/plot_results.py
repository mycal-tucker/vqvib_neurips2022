import matplotlib
import numpy as np

import src.settings as settings
from ib_color_naming.src import ib_naming_model
from src.utils.helper_fns import get_data
from src.utils.performance_metrics import PerformanceMetrics
from src.utils.plotting import plot_many_runs


def run():
    stats_metrics = []
    burnin_cutoff = 0  # How many of the first epochs to ignore due to initialization effects.
    for speaker in speaker_types:
        metrics_for_type = []
        for seed in seeds:
            metrics = PerformanceMetrics.from_file(base + speaker + str(seed))
            metrics_for_type.append(metrics)
        # And concatenate across runs
        accs = np.hstack([np.array(metric.accuracies[burnin_cutoff:]) for metric in metrics_for_type])
        informativenesses = np.hstack([np.array(metric.informativeness[burnin_cutoff:]) for metric in metrics_for_type])
        caps = np.hstack([np.array(metric.capacities[burnin_cutoff:]) for metric in metrics_for_type])
        weights = np.hstack([np.array(metric.weights[burnin_cutoff:]) for metric in metrics_for_type])

        stats_metrics.append(PerformanceMetrics(accs, caps, None, informativenesses, weights, speaker))
    plot_many_runs(stats_metrics, base, labels=labels, ib_model=ib_model, human_data=raw_data)


if __name__ == '__main__':
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}

    matplotlib.rc('font', **font)

    base = 'saved_data/'
    # List the speakers (and desired labels in the plot) you'd like to visualize.
    # E.g., for VQVIB and Onehot, speaker_types = ['VQVIB_', 'Onehot_']
    speaker_types = ['VQVIB_']
    labels = ['VQ-VIB']
    seeds = [1]  # List the seeds you want to iterate over. E.g., for seeds 0 and 1, set to [0, 1]

    ib_model = ib_naming_model.load_model()
    settings.device = 'cpu'
    raw_data, data = get_data(ib_model, 32)
    run()
