#!/usr/bin/env python

__author__ = "tchaton"

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
np.random.seed(42)

def test():
    print ("-- PCA --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = PCA(n_components=2)
    clf.fit(X_train)
    y_pred = clf.transform(X_test)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c=y_test)
    plt.show()

    '''
    -- PCA --
    '''

class PCA(BaseEstimator):
    """A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis. This class is also used throughout
    the project to plot data.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        super(PCA, self).__init__()

    def fit(self, X):
        """ Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset """
        mean_vec = X.mean(axis=0)
        cov = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        idx = np.abs(eigenvalues)
        self.best_eigenvalues = idx[:self.n_components]
        self.best_eigenvalues = eigenvectors[:, :self.n_components]
        eigenvectors = np.atleast_1d(eigenvectors)[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.best_eigenvalues)
