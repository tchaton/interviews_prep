import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from collections import Counter
import math
import cvxopt
# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False
import numpy as np
import itertools
from ..utils.utils import InputError
from ..utils.kernels import kernels
from .svm import SupportVectorMachine

def test():
    np.random.seed(42)
    print ("-- OnevsAllClassifier --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = OnevsAllClassifier(SupportVectorMachine)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    '''
    -- OnevsAllClassifier --
    Accuracy: 0.666666666667
    '''

class OnevsAllClassifier(BaseEstimator):

    def __init__(self, clf, *args):
        # Parameters
        self.clf = clf
        self.args = args
        super(OnevsAllClassifier, self).__init__()

    def _combinaison(self):
        return itertools.combinations(self.c, 2)

    def fit(self, X, y):
        self.c = np.unique(y)
        subsets = []
        self.cls = []
        for c in self.c:
            idx = np.where(y == c)
            subsets.append(X[idx])
        for combi in self._combinaison():
            i, j = combi
            x, y = subsets[i], subsets[j]
            X = np.concatenate([x, y], axis=0)
            Y = np.concatenate([i*np.ones(x.shape[0]),
                                j*np.ones(y.shape[0])], axis=0)
            #print(i, j, x.shape[0], y.shape[0], X.shape, Y.shape)
            clf = self.clf(self.args)
            clf.fit(X, Y)
            self.cls.append([clf, combi])

    def _transform_intermediate_pred(self, preds, combi):
        out = []
        for pred in preds:
            if pred < 0:
                out.append(combi[0])
            else:
                out.append(combi[1])
        return out

    def predict(self, X):
        preds_tmp = []
        preds = []
        for cl, combi in self.cls:
            preds_cl = cl.predict(X)
            preds_cl =self._transform_intermediate_pred(preds_cl, combi)
            preds_tmp.append(preds_cl)
        preds_tmp = np.array(preds_tmp)
        for pred in preds_tmp.T:
            preds.append(Counter(pred).most_common()[0][0])
        return np.array(preds)
