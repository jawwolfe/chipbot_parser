class UMAPParams:
    def __init__(self, n_neighbors, min_distance, n_components, metric, random_state):
        self.n_neighbors = n_neighbors
        self.min_distance = min_distance
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state

class HDBSCANClusters:
    def __init__(self, min_cluster_size, min_samples, cluster_metric, cluster_selection_epsilon):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_metric = cluster_metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
