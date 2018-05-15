import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
import math
import cvxopt
import numpy as np
from ..utils.utils import InputError
from ..utils.kernels import kernels

def test():
    np.random.seed(42)
    print ("-- SVM Classifier --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = SVMClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    '''
    -- SVM Classifier --
    ('Accuracy:', 0.983333333333)
    '''

class SVMClassifier(BaseEstimator):

    def __init__(self, C=1, kernel='rbf', power=4, gamma=None, coef=4, *args, **kwargs):

        # Parameters
        self.C = C
        self.create_kernel(kernel)
        self.power = power
        self.gamma = gamma
        self.coef = coef
        super(SVMClassifier, self).__init__()

        # Attributes
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vectors_labels = None
        self.intercept = None

    def create_kernel(self, kernel):
        kernels_names = ['rbf', 'poly', 'linear']
        if kernel in kernels_names:
            self.kernel = kernel
            self.kernel_func = kernels[kernel]
        else:
            raise InputError(str(kernel)+' not in '+str(kernels))

    def fit(self, X, y):
        n_samples, features = np.shape(X)

        if not self.gamma:
            self.gamma = 1/float(features)

        self.kernel = self.kernel_func(power=self.power,
                                       coef=self.coef,
                                       gamma=self.gamma)

        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        #######################################################################################
        # TAKEN FROM ML FROM SCRATCH - NOT THE TIME TO GET DEEPER IN THE OPTIMIZATION LIBRARY #
        #######################################################################################

        ### CHECK http://goelhardik.github.io/2016/11/28/svm-cvxopt/ FOR A CLEAR TUTORIAL


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
        idx = lagr_mult > 1e-7
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
