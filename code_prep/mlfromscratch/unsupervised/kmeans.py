#!/usr/bin/env python

__author__ = "tchaton"


''' Implementation of Kmeans algorithm '''

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

np.random.seed(42)


def test():
    print ("-- KMEANS Classification --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    clf = KmeansEstimator(3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    '''
    -- KMEANS Classification --
    ('Accuracy:', 0.6666666666666666)

    '''

class KmeansEstimator(BaseEstimator):

    def __init__(self, nb_clusters, criteria_stop=10e-7, update_max_step = 10e4):
        self.nb_clusters = nb_clusters
        self.criteria_stop = criteria_stop
        self.update_max_step = update_max_step


    def init_clusters(self):
        self.clusters = []
        L = len(self.X)
        ratio = L/10.
        for _ in range(self.nb_clusters):
            indexes = np.random.choice(range(L), int(ratio), replace=False)
            sub = self.X[indexes]
            mean, std = np.mean(sub, axis=0), np.std(sub, axis=0)
            rand = np.random.normal(mean, std, sub[0].shape)
            self.clusters.append(rand)
        self.clusters = {str(index):centroid for index, centroid in enumerate(self.clusters)}

    def distance(self, x, y, metric='euclidean'):
        if metric == 'euclidean':
            return np.sum(np.sqrt((x-y)**2))


    def loop(self):
        self.previous_clusters = {key:self.clusters[key] for key in self.clusters.keys()}
        dist_m = np.zeros((len(self.X), len(self.clusters.items())))
        for i, x in enumerate(self.X):
            for centroid in self.clusters.items():
                dist = self.distance(x, centroid[1])
                dist_m[i, int(centroid[0])] = dist


        linked2c = np.argmin(dist_m, axis=-1)
        for i in range(self.nb_clusters):
            indexes = np.where(linked2c == i)
            sub = self.X[indexes]
            mean = np.mean(sub, axis=0)
            if not np.isnan(mean).any():
                self.clusters[str(i)] = mean

        diff = 0
        for key in self.clusters.keys():
            diff+=(self.distance(self.clusters[key], self.previous_clusters[key]))
        #print(diff)
        if diff > self.criteria_stop:
            return False
        else:
            return True

    def predict(self, X):
        dist_m = np.zeros((len(X), len(self.clusters.items())))
        for i, x in enumerate(X):
            for centroid in self.clusters.items():
                dist = self.distance(x, centroid[1])
                dist_m[i, int(centroid[0])] = dist
        linked2c = np.argmin(dist_m, axis=-1)
        return linked2c

    def plot(self):
        X = np.copy(self.X)
        for i in range(self.nb_clusters):
            X = X+[self.clusters[str(i)]]
        tnse = TSNE(n_components=2)
        projected = tnse.fit_transform(X)
        Y = self.predict(self.X)
        plt.scatter(projected[:-self.nb_clusters, 0], projected[:-self.nb_clusters, 1], c = Y[:-self.nb_clusters])
        plt.scatter(projected[-self.nb_clusters:, 0], projected[-self.nb_clusters:, 1], color='red')
        plt.show()

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.init_clusters()
        activateBreak = False
        cnt = 0
        #self.plot()
        while cnt < self.update_max_step and not activateBreak :
            activateBreak = self.loop()
            #self.plot()
            #print(activateBreak, cnt)
            cnt+=1






