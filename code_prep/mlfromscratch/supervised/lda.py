#!/usr/bin/env python

__author__ = "tchaton"

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from ..utils.operations import euclidean

np.random.seed(42)

def test():
    print ("-- LDA --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = LDA(n_components=2)
    clf.fit(X_train, y_train)
    y_pred, hs = clf.predict(X_test, with_plot=True)
    plt.scatter(hs[:, 0], hs[:, 1], c=y_test)
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    '''
    -- LDA --
    Accuracy: 1.0
    '''

class LDA():
    """The Linear Discriminant Analysis classifier, also known as Fisher's linear discriminant.
    Can besides from classification also be used to reduce the dimensionaly of the dataset.
    Reference : https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
    """
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        self.X = X
        self.y = y
        self._compute_within_class_scatter_matrices()
        self._compute_between_class_scatter_matrices()
        self._compute_eigenvalues()

    def _compute_within_class_scatter_matrices(self):
        self.c = np.unique(self.y)
        self.within_container = {}
        self.Sw = 0
        for c in self.c:
            idx_c = np.where(c == self.y)
            sub_c = self.X[idx_c]
            N_c = len(idx_c[0])
            mean_c = np.mean(sub_c, axis=0)
            scatter_mat = np.sum([(x - mean_c)[np.newaxis, ...].T.dot((x - mean_c)[np.newaxis, ...]) for x in sub_c], axis=0)
            self.within_container[str(c)] = [N_c, mean_c, scatter_mat]
            self.Sw += (1/(N_c-1))*scatter_mat
        self.within_container[str(-1)] = [len(self.y),
                              np.mean(self.X, axis=0),
                              np.sum([(x - mean_c)[np.newaxis, ...].T.dot((x - mean_c)[np.newaxis, ...]) for x in self.X], axis=0)]

    def _compute_between_class_scatter_matrices(self):
        mat = 0
        mean_d = self.within_container[str(-1)][1]
        for c in self.c:
            N_c, mean_c, _ = self.within_container[str(c)]
            diff = (mean_c - mean_d)[np.newaxis, ...]
            mat+= N_c * diff.T.dot(diff)
        self.Sb = mat
    def _compute_eigenvalues(self):
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(self.Sw).dot(self.Sb))
        self.eigenvectors = eig_vecs[:self.n_components]
        for c in self.c:
            mean_c = self.within_container[str(c)][1]
            self.within_container[str(c)].append(mean_c.dot(self.eigenvectors.T))

    def predict(self, X, with_plot=False):
        y_pred = []
        hs = []
        for sample in X:
            h = sample.dot(self.eigenvectors.T)
            if with_plot:
                hs.append(h)
            pred = np.argmin([euclidean(h, self.within_container[str(c)][-1]) for c in self.c])
            y_pred.append(pred)
        if with_plot:
            return y_pred, np.array(hs)
        else:
            return y_pred
