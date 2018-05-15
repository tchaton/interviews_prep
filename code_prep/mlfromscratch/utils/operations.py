import numpy as np
from collections import Counter

class Metrics:

    def __init__(self, X):
        self.X = X

    def set_metrics(self, metrics):
        self.metrics = metrics
        if metrics == 'correlation':
            self.func_metric = correlation
        elif metrics == 'euclidean':
            self.func_metric = euclidean

    def calculate_distance_matrix(self, Z, metrics='euclidean'):
        self.set_metrics(metrics)
        x_samples, features = np.shape(self.X)
        z_samples, feature = np.shape(Z)
        dist = np.zeros((x_samples, z_samples))
        for i in range(x_samples):
            for j in range(z_samples):
                dist[i, j] = self.func_metric(self.X[i], Z[j])
        self.dist = np.array(dist).T

    def calculate_preds(self, Y, k):
        func = None
        closest = np.argsort(self.dist, axis=-1)
        if self.metrics == 'euclidean':
            preds = closest[:, :k]
        elif self.metrics == 'correlation':
            preds = closest[:, -k:]
        preds = np.take(Y, preds)
        preds = np.array([Counter(e).most_common(1) for e in preds])
        return preds[:, 0, 0]

def correlation(x, z):
    pass

def euclidean(x, z):
    return np.sqrt(np.sum((x-z)**2))
