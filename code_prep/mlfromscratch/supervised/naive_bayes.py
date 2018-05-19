#!/usr/bin/env python

__author__ = "tchaton"

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..utils.operations import mean, std

np.random.seed(42)


def test():
    print ("-- Naive Bayes --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)


    print ("Accuracy:", accuracy)
    '''
    -- Naive Bayes --
    Accuracy: 0.983333333333

    '''

class NaiveBayes(BaseEstimator):
    '''
    Fundamental Bayes equation : p(A|B) = p(B|A)*p(A)/p(B)
    '''

    def __init__(self):
        super(NaiveBayes, self).__init__()

    def _summarize_class(self):
        self.container = {}
        self.c = np.unique(self.y)
        for u in self.c:
            idxu = np.where(u == self.y)
            sub = self.X[idxu]
            self.container[str(u)] = [len(idxu[0])/float(len(self.y)), mean(sub), std(sub)]

    def _calculate_posterior(self, x, mean, var, eps=1e-4):
        coeff = (1/(var*np.sqrt(2*np.pi)))
        exp = np.exp(-(x-mean)**2/(2*var + eps))
        return coeff*exp

    def fit(self, X, y):
        self.X = X
        self.y = y
        self._summarize_class()

    def predict(self, X):
        preds = []
        for x in X:
            h = []
            for u in self.c:
                prior, mean, var = self.container[str(u)]
                posterior = self._calculate_posterior(x, mean, var)
                h.append(prior*np.product(posterior)) # NAIVE BAYES : FEATURES INDEPENDANCE
            preds.append(np.argmax(h))
        return preds
