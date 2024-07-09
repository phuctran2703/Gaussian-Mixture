import numpy as np
from scipy.stats import multivariate_normal
from k_mean import *

class GaussianMixtureModel:
    def __init__(self, num_components=1, max_iter=100, tol=1e-3):
        self.num_components = num_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_param(self, X):
        num_samples, num_features = X.shape
        # indices = np.random.choice(num_samples, self.num_components, replace=False)
        # self.means = X[indices]
        self.means = KMeansModel(num_clusters = self.num_components).fit(X).means
        self.covs = np.array([np.eye(num_features) for _ in range(self.num_components)])
        self.coefs = np.ones(self.num_components) / self.num_components

        self.log_likelihoods = []
        log_likelihood = self.compute_log_likelihood(X)
        self.log_likelihoods.append(log_likelihood)

        return self

    def e_step(self, X):
        num_samples = X.shape[0]
        responsibilities = np.zeros((num_samples, self.num_components))

        for k in range(self.num_components):
            normal = multivariate_normal(mean=self.means[k], cov=self.covs[k])
            responsibilities[:, k] = self.coefs[k] * normal.pdf(X)

        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        num_samples = X.shape[0]
        N_k = responsibilities.sum(axis=0)

        self.means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]

        for k in range(self.num_components):
            delta = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * delta.T, delta) / N_k[k]

        self.coefs = N_k / num_samples

    def compute_log_likelihood(self, X):
        num_samples = X.shape[0]
        log_likelihood = 0.0

        for n in range(num_samples):
            prob = 0.0
            for k in range(self.num_components):
                normal = multivariate_normal(mean=self.means[k], cov=self.covs[k])
                prob += self.coefs[k] * normal.pdf(X[n])
            log_likelihood += np.log(prob)

        print("log_likelihood:", log_likelihood)

        return log_likelihood

    def fit(self, X):
        self.initialize_param(X)

        for i in range(self.max_iter):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)

            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)

            if i > 0 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break

        return self
    
    def predict(self, test_data):
        return np.argmax(self.e_step(test_data), axis=1)