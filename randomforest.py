from decisiontree import DecisionTree
import numpy as np


class RandomForest:
    """Ensemble learning with bagging on samples
        and selecting subsets of features.

        Attributes:
                fc (int): number of subset features selected.
                tree_num (int): number of decision trees in forest.
                trees (dict): dictionary to store trees.
                features (dict): dictionary to store selected features.
                max_depth (int): maximum depth of each decision tree.
                eval_func (string): evaluation criteria, either gini impurity or entropy.
    """

    def __init__(self, feature_count=1, tree_num=10,
                 depth=10, min_improv=1e-8, eval_func="gini_impurity"):
        """Initialize randomforest.

        Args:
                feature_count (int): number of subset features selected.
                tree_num (int): number of decision trees in forest.
                depth (int): maximum depth of each decision tree.
                min_improv (float): minimum improvement in gini impurity/entropy.
                                eval_func (string): evaluation criteria, either gini impurity or entropy.
        """
        self.fc = feature_count
        self.tree_num = tree_num
        self.trees = {}  # dictionary to store trees
        self.features = {}  # dictionary to store selected
        self.max_depth = depth
        self.min_improv = min_improv
        if eval_func not in ['gini_impurity', 'entropy']:
            raise "Undefined evaluation criteria, choose either gini_impurity or entropy."
        self.eval_func = eval_func

    def fit(self, X, y):
        """Build multiple trees based on training data.

        Args:
            X (numpy array): sample in shape [n x d], where n is
            number of samples and d is number of features.
            y (numpy array): sample labels in shape [n].
        """
        n, d = X.shape
        for i in range(self.tree_num):
            # draws random subset of features
            features = np.random.choice(d, self.fc, replace=False)
            tree = DecisionTree(
                self.max_depth, self.min_improv, self.eval_func)
            samples = np.random.choice(n, n, replace=True)
            X_train = X[:, features][samples, ]
            y_train = y[samples]
            tree.fit(X_train, y_train)

            self.features[i] = features
            self.trees[i] = tree

    def predict(self, X):
        """Predict labels given sample dataset X, use majority rule.

        Args:
            X (numpy array): samples in shape [n x d], where n is number of samples,
                                and d is number of features.

        Returns:
            y (numpy array): predicted label of shape [n], each entry is binary.
        """
        y = []
        for i, tree in self.trees.items():
            features = self.features[i]
            X_test = X[:, features]
            y.append(tree.predict(X_test))
        sums = np.sum(np.stack(y, axis=1), axis=1)
        return (sums > (self.tree_num / 2)) * 1
