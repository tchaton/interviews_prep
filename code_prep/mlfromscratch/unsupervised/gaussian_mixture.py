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
from collections import Counter
from matplotlib import pyplot as plt

np.random.seed(42)


def test():
    print ("-- GaussianMixtureModel Classification --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = GaussianMixtureModel(3)
    clf.fit(X_train)
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

    def _sample_centroids(self, init_prev=False):
        if not init_prev:
            n_samples, features = np.shape(self.X)
            n = n_samples//self.nb_clusters
            samples = np.random.choice(range(n_samples), n)
            d = {}
            std = np.std(self.X[samples], axis=0)
            d['mean'] =  np.mean(self.X[samples], axis=0)# + np.random.normal(0, std, self.X[0].shape)
            centered = (self.X[samples] - d['mean'])
            d['cov'] = centered.T.dot(centered)
            return d
        else:
            d = {}
            d['mean'] = np.zeros_like(self.X[0])
            X = self.X[0][np.newaxis,...]
            d['cov'] = np.zeros_like(X.T.dot(X))
            return d

    def _init_cluster(self):
        self.prev_centroids = [self._sample_centroids(init_prev=True) for _ in range(self.nb_clusters)]
        self.centroids = [self._sample_centroids() for _ in range(self.nb_clusters)]
        #print(self.prev_centroids)
        #print(self.centroids)
        self.priors = np.ones((self.nb_clusters))/float(self.nb_clusters)

    def _get_by_key(self, arr, key):
        return np.array([a[key] for a in arr])

    def _is_tolerance_reached(self):
        dist = np.mean(np.linalg.norm(self._get_by_key(self.centroids, 'mean') - self._get_by_key(self.prev_centroids, 'mean')))
        print(dist)
        if dist < self.tolerance:
            return True
        else:
            return False

    def _gaussian_likelihood(self, sample, cluster):
        '''
        Evaluation of the Gaussian PDF
        Reference : https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        '''
        mean, cov = cluster['mean'], cluster['cov']
        factor = np.power(np.sqrt(2*np.pi*np.linalg.det(cov)), -1/2)
        X = (sample - mean)[..., np.newaxis]
        coeff = X.T.dot(np.linalg.inv(cov)).dot(X)
        exp = np.exp((-1/2)*coeff) # (1, 4) * (4, 4) * (4, 1) = (1 , 1)
        return factor*exp

    def _get_likehoods(self):
        samples_likelihoods = []
        for sample in self.X:
            likelihoods = []
            for centroid in self.centroids:
                likelihoods.append(self._gaussian_likelihood(sample, centroid))
            samples_likelihoods.append(likelihoods)
        return np.squeeze(samples_likelihoods)

    def _expectation(self):
        '''Calculate the responsability of each cluster'''
        weighted_likehoods = self._get_likehoods() * self.priors
        for i in range(len(weighted_likehoods)):
            weighted_likehoods[i]/=np.sum(weighted_likehoods[i])
        selected_clusters = np.array([np.random.choice(range(self.nb_clusters), p=wl) for wl in weighted_likehoods])
        self.selected_clusters = selected_clusters
        
    def _maximization(self):
        pass

    def fit(self, X):
        '''
        Run GMM
        '''
        self.X = X
        # INIT clusters
        self._init_cluster()
        # RUN GaussianMixture
        for i in range(self.max_iterations):
            print(i)
            if not self._is_tolerance_reached():
                self._expectation()
                self._maximization()
            break

    def predict(self, X):
        '''
        Return clusters attribution indices
        '''
