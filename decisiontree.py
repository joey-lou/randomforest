"""
Simple binary decision tree implementation
"""
import numpy as np
from collections import deque


class Node:
    """Node for decision tree

    Attributes:
        feature (int): index of feature for split at the node.
        thresh (float): threshold for splitting at node.
        left (Node): left node, could be None if current node is leaf.
        right (Node): right node, could be None if current node is leaf.
        depth (int): current depth of node.
        prob (float): [0,1] for probability of positive.
    """

    def __init__(self):
        """Initialize empty node with Nones.
        """
        self.feature = None
        self.thresh = None
        self.left = None
        self.right = None
        self.depth = None
        self.prob = None

    def __repr__(self):
        return str([self.feature, self.thresh, self.label])


class DecisionTree:
    """Decision tree for binary classification only.

    Attributes:
        root (Node): root of decision tree.
        max_depth (int): maximum depth set for decision tree.
        min_improv (float): minimum improvement in gini impurity/entropy.
        eval_func (string): evaluation criteria, either gini impurity or entropy.
    """

    def __init__(self, depth=10, min_improv=1e-8, eval_func="gini_impurity"):
        """Initialize decision tree with root node.

        Args:
            depth (str): maximum depth for decision tree, default = 10.
            min_improv (float): minimum improvement in information gain needed
                                for split, default = 1e-6.
            eval_func (string): evaluation criteria, "gini_impurity" or "entropy"
                                default = "gini_impurity"
        """
        self.root = Node()  # create root node
        self.max_depth = depth
        self.min_improv = min_improv
        if eval_func not in ['gini_impurity', 'entropy']:
            raise "Undefined evaluation criteria, choose either gini_impurity or entropy."
        self.eval_func = eval_func

    def entropy(self, pos, neg):
        """Calculates entropy given labels.

        Args:
            pos (int): number of positive labels (count of 1s)
            neg (int): number of negative labels (count of 0s)

        Returns:
            float: entropy score.
        """
        res = 0.0
        if neg != 0:
            n = neg / (pos + neg)
            res -= n * np.log2(n)
        if pos != 0:
            p = pos / (pos + neg)
            res -= p * np.log2(p)
        return res

    def gini_impurity(self, pos, neg):
        """Calculates gini impurity given labels.

        Args:
            pos (int): number of positive labels (count of 1s)
            neg (int): number of negative labels (count of 0s)

        Returns:
            float: gini impurity score.
        """
        if pos == neg == 0:
            return 0.0
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return p * (1 - p) + n * (1 - n)

    def purity_improv(self, X, y, thresh):
        """Calculates decrease in gini impurity/ entropy from split at thresh.

        Args:
            X (numpy array): single feature sample data [n].
            y (numpy array): sample labels in shape [n].
            thresh (float): threshold for splitting of given feature.

        Returns:
            float: decrease in gini impurity or entropy.
        """
        if self.eval_func == "gini_impurity":
            func = self.gini_impurity
        else:
            func = self.entropy
        root = func(y)
        idx = (X <= thresh)
        N_l = np.sum(idx) / len(idx)
        N_r = 1 - N_l
        left = func(y[idx])
        right = func(y[~idx])
        return root - N_l * left - N_r * right

    def split(self, X, y, idx, thresh):
        """Splits given dataset and label into two based on threshold
            and feature index.

        Args:
            X (numpy array): sample in shape [n* x d], where n* is
            number of samples in parent node and d is number of features.
            y (numpy array): sample labels in shape [n*].
            idx (int): index for split feature.
            thresh (float): value for splitting that feature.

        Returns:
            X_l (numpy array): samples that will split to left leaf.
            y_l (numpy array): labels that will split to left leaf.
            X_r (numpy array): samples that will split to right leaf.
            y_r (numpy array): labels that will split to right leaf.
        """
        # assume [sample x feature] indexing
        neg = X[:, idx] <= thresh
        return X[neg, ], y[neg], X[~neg, ], y[~neg]

    def __segmenter(self, X, y, func):
        """Find best feature splitting given dataset and label.

        Takes in dataset and find best feature and threshold
        for splitting, which results in most improvement in purity.

        Args:
            X (numpy array): sample in shape [n x d], where n is
            number of samples and d is number of features.
            y (numpy array): sample labels in shape [n].
            func (class method): evaluation function for improvement,
                                gini_impurity or entropy.
        Returns:
            ridx (int): index of selected split feature.
            best_thresh (float): value for splitting.
            improvement (float): decrease in evaluation score.
        """
        best = float('inf')
        ridx = 0
        best_thresh = 0
        m = len(y)
        total_pos = np.sum(y)
        # calculate entropy before split
        before = func(total_pos, m - total_pos)
        current = before
        iterator = np.arange(m)
        for j in range(X.shape[1]):
            # loop over all features
            X_feature = X[:, j]
            idx = np.argsort(X_feature)
            X_feature = X_feature[idx]
            y_sorted = y[idx]
            # re-initialize assignment for new feature
            l_pos, l_neg = 0, 0
            r_pos = total_pos
            r_neg = m - r_pos
            # iterate over sorted feature array
            for thresh, sign, i in zip(X_feature, y_sorted, iterator):
                if sign == 1:
                    l_pos += 1
                    r_pos -= 1
                else:
                    l_neg += 1
                    r_neg -= 1

                current = i / m * func(l_pos, l_neg) + \
                    (m - i) / m * func(r_pos, r_neg)

                # minimize gini impurity
                if current < best:
                    best = current
                    best_thresh = thresh
                    ridx = j

        return ridx, best_thresh, before - current

    def fit(self, X, y):
        """Recursively form decision tree based on training data.

        Calls dfs_split method to recursively form decision tree using training data.
        Starts from root node initialized at formation of tree.

        Args:
            X (numpy array): sample in shape [n x d], where n is
            number of samples and d is number of features.
            y (numpy array): sample labels in shape [n].
        """
        if self.eval_func == "gini_impurity":
            self.__dfs_split(X, y, self.root, 0, self.gini_impurity)
        else:
            self.__dfs_split(X, y, self.root, 0, self.entropy)

    def __dfs_split(self, X, y, node, depth, func):
        """Split a given node in decision tree based on input data.

        Recrusively split node till no more improvements can be made or
        maximum tree depth has been reached.

        Args:
            X (numpy array): sub-sample in shape [n* x d*], where n*
            is number of samples and d* is number of features.
            y (numpy array): sub-sample labels in shape [n*].
            node (Node): current node for splitting.
            depth (int): current depth at point of splitting.
            func (instance method): calculates either entropy or gini impurity

        """
        idx, thresh, improvement = self.__segmenter(X, y, func)
        # update current node
        node.feature = idx
        node.thresh = thresh
        node.depth = depth

        # stop splitting if criteria met
        if depth == self.max_depth or improvement < self.min_improv:
            node.prob = sum(y) / len(y)
            return

        # move on to left and right leaves
        node.left = Node()
        node.right = Node()
        X_left, y_left, X_right, y_right = self.split(X, y, idx, thresh)
        self.__dfs_split(X_left, y_left, node.left, depth + 1, func)
        self.__dfs_split(X_right, y_right, node.right, depth + 1, func)

    def predict(self, X):
        """Predict label given sample dataset X.

        Predict labels given dataset after training of decision tree.

        Args:
            X (numpy array): samples in shape [n x d], where n is number of samples,
                                and d is number of features.

        Returns:
            y (numpy array): predicted label of shape [n], {0,1}.
        """
        probs = self.predict_prob(X)
        return (probs > 0.5) * 1

    def predict_prob(self, X):
        """Predict probabilities of positive given sample dataset X.

        Predict labels given dataset after training of decision tree.

        Args:
            X (numpy array): samples in shape [n x d], where n is number of samples,
                                and d is number of features.

        Returns:
            y (numpy array): predicted probabilities of shape [n], each [0,1].
        """
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # call predict_single for individual sample
            y[i] = self.__predict_single(X[i, ], self.root)
        return y

    def __predict_single(self, x, node):
        """Predict a probability given single sample using recursion.

        Args:
            x (numpy array): single sample, array of size d (number of features).
            node (Node): current node for evaluation.

        Returns:
            prob (float): probability of positive [0,1].
        """
        if node.prob is not None:
            return node.prob
        if x[node.feature] <= node.thresh:
            return self.__predict_single(x, node.left)
        else:
            return self.__predict_single(x, node.right)

    def __repr__(self):
        """BFS printout of decision tree."""

        queue = deque([self.root])
        depth = 0
        out = ""
        while queue:
            node = queue.popleft()
            if not node:
                out += "[None]"
            else:
                if node.depth != depth:
                    out += "\n"
                    depth = node.depth
                if node.prob is not None:
                    out += "(" + str(node.prob) + ")"
                else:
                    out += "[{:d},{:3.2f}]".format(node.feature, node.thresh)
                    queue.append(node.left)
                    queue.append(node.right)
        return out
