import numpy as np

class KMeansModel:
    def __init__(self, num_clusters=1, max_iter=100, tol=1e-4):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol

    def initialize_params(self, X):
        np.random.seed(0)
        num_samples = X.shape[0]
        random_indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        self.means = X[random_indices]
        self.in_var = np.zeros((num_samples, self.num_clusters))

    def fit(self, X):
        num_samples, num_features = X.shape
        self.initialize_params(X)

        for iteration in range(self.max_iter):
            # E-step
            for i in range(num_samples):
                distances = np.linalg.norm(X[i] - self.means, axis=1)
                cluster = np.argmin(distances)
                self.in_var[i, cluster] = 1

            # M-step
            new_means = np.zeros((self.num_clusters, num_features))
            for k in range(self.num_clusters):
                cluster_points = X[self.in_var[:, k] == 1]
                if len(cluster_points) > 0:
                    new_means[k] = cluster_points.mean(axis=0)
                else:
                    new_means[k] = self.means[k]

            # Check for convergence
            if np.linalg.norm(self.means - new_means) < self.tol:
                break
            self.means = new_means

        return self

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)
        for i in range(num_samples):
            distances = np.linalg.norm(X[i] - self.means, axis=1)
            closest_cluster = np.argmin(distances)
            predictions[i] = closest_cluster
        return predictions
