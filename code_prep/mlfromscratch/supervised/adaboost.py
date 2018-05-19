#!/usr/bin/env python

__author__ = "tchaton"

''' Implementation Adabost Classifier '''

import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
import math
import numpy as np

np.random.seed(42)
### Code Reference : https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py

def test():
    print ("-- Adaboost Classifier --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    weaks = [WeakClassifier() for _ in range(10)]
    # Verify weak classifiers simulators
    for i, weak in enumerate(weaks):
        pred = weak.predict(X_test, y_test)
        accuracy = accuracy_score(y_test, pred)
        print('Accuracy weak classifier '+str(i)+' :'+ str(accuracy))

    clf = Adaboost()
    clf.fit(X_train, y_train, cls=weaks)
    y_pred = clf.predict(X_test, y_test) # WILL BE USED TO SIMULATE WEAK CLASSIFIERS
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    '''
    -- Adaboost Classifier --
    Accuracy weak classifier 0 :0.7
    Accuracy weak classifier 1 :0.533333333333
    Accuracy weak classifier 2 :0.766666666667
    Accuracy weak classifier 3 :0.7
    Accuracy weak classifier 4 :0.816666666667
    Accuracy weak classifier 5 :0.733333333333
    Accuracy weak classifier 6 :0.8
    Accuracy weak classifier 7 :0.9
    Accuracy weak classifier 8 :0.766666666667
    Accuracy weak classifier 9 :0.9
    Accuracy: 1.0
    '''
class WeakClassifier(BaseEstimator): # SIMULATE WEAK CLASSIFIER

    def __init__(self):
        super(WeakClassifier, self).__init__()
        self.seed = 42

    def predict(self, X, Y, eps=0.1): # USAGE y
        preds = []
        uni = np.unique(Y)
        self.score = np.random.uniform(0.6, 0.9) # Better than random guessing
        for y in Y:
            if np.random.uniform(0, 1) < self.score:
                preds.append(y)
            else:
                y = np.random.choice([x for x in uni if x != y])
                preds.append(y)
        acc = accuracy_score(Y, preds)
        self.error = 1 - acc
        return preds

def ln2(x):
    return np.log(x) / np.log(2)

class Adaboost(BaseEstimator):

    '''
    Already trained Classifiers
    Reference : http://rob.schapire.net/papers/explaining-adaboost.pdf
    Reference :http://mccormickml.com/2013/12/13/adaboost-tutorial/
    '''

    def __init__(self, iteration=50, learning_rate=1):
        self.iteration = iteration
        self.learning_rate = learning_rate
        super(Adaboost, self).__init__()

    def _calculate_weights(self):
        L = len(self.cls)
        self.weights = np.ones((L,))/float(L)
        for index, cl in enumerate(self.cls):
            error = cl.error
            estimator_weights = self.learning_rate * (1/2) * (ln2(1-error)-ln2(error))
            self.weights[index]*=estimator_weights
        #for w, cl in zip(self.weights, self.cls):
        #    print(w, cl.score)

    def fit(self, X, Y, cls=[]):
        self.X = X
        self.Y = Y
        self.cls = cls
        self._calculate_weights()


    def predict(self, X, y_test):
        preds = np.zeros((X.shape[0], len(np.unique(y_test))))
        tmp_preds = np.array([cl.predict(X, y_test) for cl in self.cls]).T
        for index, pred in enumerate(tmp_preds):
            for index_cl, pred_cl in enumerate(pred):
                preds[index, pred_cl]+= 1 * self.weights[index_cl]
        return np.argmax(preds, axis=-1)
