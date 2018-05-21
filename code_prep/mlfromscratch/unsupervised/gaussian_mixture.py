#!/usr/bin/env python

'''
Python Implementation : GaussianMixtureModel
'''

__author__ = "tchaton"


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

np.random.seed(42)


def test():
    print ("-- GaussianMixtureModel Classification --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = GaussianMixtureModel(3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    '''
    -- GaussianMixtureModel Classification --
    ('Accuracy:', 0.6666666666666666)

    '''



class GaussianMixtureModel():
    """A probabilistic clustering method for determining groupings among data samples.
    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    tolerance: float
        If the difference of the results from one iteration to the next is
        smaller than this value we will say that the algorithm has converged.
    """

    def __init__(self, nb_clusters=2, max_iterations=2000, tolerance=1e-7):
        # Parameters
        self.nb_clusters = nb_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Attributes

    def _init_cluster(self):
        pass



    def predict(self, X):
        '''
        Run GMM and return clusters attribution indices
        '''
        # INIT clusters
        self._init_cluster()
