import pickle


class PerformanceMetrics:
    def __init__(self, accuracies, capacities, recons, comm_accs, gnids, weights, clusters, label):
        self.accuracies = accuracies
        self.capacities = capacities
        self.recons = recons
        self.comm_accs = comm_accs
        self.gnids = gnids
        self.weights = weights
        self.clusters = clusters
        self.label = label

    def to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            loaded = pickle.load(file)
        return loaded
