import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from ..operations import Metrics
import math
import numpy as np

def test():
    np.random.seed(42)
    print ("-- KNeighborsClassifier --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, metrics='euclidean')
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    '''
    -- KNeighborsClassifier  Tree --
    ('Accuracy:', 0.983333333333)
    '''

class KNeighborsClassifier(BaseEstimator):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        super(KNeighborsClassifier, self).__init__()

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.metrics = Metrics(X)

    def predict(self, X, metrics='euclidean'):
        self.metrics.calculate_distance_matrix(X, metrics=metrics)
        return self.metrics.calculate_preds(self.Y, self.n_neighbors)
