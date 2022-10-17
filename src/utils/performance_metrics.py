import pickle


class PerformanceMetrics:
    def __init__(self, accuracies, capacities, recons, informativeness, weights, label):
        self.accuracies = accuracies
        self.capacities = capacities
        self.recons = recons
        self.informativeness = informativeness
        self.weights = weights
        self.label = label

    def to_file(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(path):
        with open(path, 'rb') as file:
            loaded = pickle.load(file)
        return loaded
