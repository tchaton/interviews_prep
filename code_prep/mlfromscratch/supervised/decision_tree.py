import os, sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
import math
import numpy as np

### Code Reference : https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/decision_tree.py

def test():
    print ("-- Classification Tree --")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)


    print ("Accuracy:", accuracy)

def exp_dim(arr):
    return np.expand_dims(arr, axis=-1)

def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])

class DecisionNode:
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None,
                       threshold=None,
                       value=None,
                       true_branch=None,
                       false_branch=None):

        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(BaseEstimator):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """
    def __init__(self, min_samples_split=2,
                       min_impurity=10e-7,
                       max_depth=float('inf'),
                       loss=None):
        # PARAMETERS
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss

        # ATTRIBUTES
        self.root = None # Root
        self._impurity_calculation = None # classif : gain, regr : variance reduction
        self._leaf_value_calculation = None # One hot
        self.one_dim = None # if Gradient Boost

    def fit(self, x, y=None):
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(x, y)
        self.loss = None

    def _build_tree(self, x, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""

        largest_impurity = 0
        best_criteria = None

        # Augment y
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Concatenate X and Y
        Xy = np.concatenate([x, y], axis=1)

        n_samples, n_features = np.shape(Xy)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:

            for features_index in range(n_features):

                features_values = np.expand_dims(Xy[:, features_index], axis=1)
                unique_values = np.unique(features_values)

                # Try all possible values as a split
                for threshold in unique_values:


                    Xy1, Xy2 = divide_on_feature(Xy, features_index, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)
                        print(impurity, current_depth)
                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": features_index, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

                if largest_impurity > self.min_impurity:
                    # Build subtrees for the right and left branches
                    true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
                    false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
                    return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                        "threshold"], true_branch=true_branch, false_branch=false_branch)
            # is_leaf
            value = self._leaf_value_calculation(y)
            return DecisionNode(value=value)

    def predict(self, X):
        def _predict(x, node):
            if node:
                if node.value == None:
                    feature_i = node.feature_i
                    threshold = node.threshold
                    if x[feature_i] < threshold:
                        node = node.false_branch
                    else:
                        node = node.true_branch
                    return _predict(x, node)
                else:
                    return int(node.value)

        preds = [_predict(x, self.root) for x in X]
        return preds

def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    y = y.flatten()
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / float(len(y))
        entropy += -p * log2(p)
    return entropy

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
                              calculate_entropy(y1) - (1 - p) * \
                                                      calculate_entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        print(np.unique(y))
        most_common = None
        max_count = 0
        for label in np.unique(y):
        # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, x, y=None):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(x, y)
        self.loss = None