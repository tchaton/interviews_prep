import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from collections import Counter
from sklearn.svm import SVC
import math
import cvxopt
# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False
import numpy as np
import itertools
from ..utils.utils import InputError
from ..utils.kernels import kernels

rbf_kernel = kernels['rbf']

def generate_subset(X, y):
    classes = np.unique(y)
    subsets = []
    for c in classes:
        idx = np.where(y == c)
        sub = X[idx]
        subsets.append([sub, c*np.ones(sub.shape[0])])
    for combi in itertools.combinations(classes, 2):
        print(combi)
        i, j = combi
        x, y = subsets[i], subsets[j]
        X = np.concatenate([x[0], y[0]], axis=0)
        Y = np.concatenate([x[1], y[1]], axis=0)
        yield X, Y

def test():
    np.random.seed(42)
    print ("-- SVM Classifier --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    for train, test in zip(generate_subset(X_train, y_train), generate_subset(X_test, y_test)):
        clf = SupportVectorMachine()
        X_train, y_train = train
        X_test, y_test= test
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print ("Accuracy:", accuracy)

        clf_sklearn = SVC()
        clf_sklearn.fit(X_train, y_train)
        y_pred = clf_sklearn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print ("Accuracy:", accuracy)

    '''
    -- SVM Classifier -- ERROR IN THE SOLUTION COMPARED TO SKLEARN - Need investigation
    (0, 1)
    (0, 1)
    Accuracy: 0.547619047619
    Accuracy: 1.0
    (0, 2)
    (0, 2)
    Accuracy: 0.560975609756
    Accuracy: 1.0
    (1, 2)
    (1, 2)
    Accuracy: 0.513513513514
    Accuracy: 1.0

    '''

class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    Parameters:
    -----------
    C: float
        Penalty term.
    kernel: function
        Kernel function. Can be either polynomial, rbf or linear.
    power: int
        The degree of the polynomial kernel. Will be ignored by the other
        kernel functions.
    gamma: float
        Used in the rbf kernel function.
    coef: float
        Bias term used in the polynomial kernel function.
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            gamma=self.gamma)

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-11
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
