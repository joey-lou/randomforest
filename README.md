# Random Forest from Stratch
While existing packages such as Sklearn are readily available, I wanted to implement decision trees and Random Forest from the ground up to gain better understandings. Here we have built simple decision trees for binary classifcation problems, and a random forrest classifier built on top.


## Set-up
To use this package, only numpy is required. If not installed run: `pip install numpy` in terminal.

To run comparisons with Scikit-learn in sample_test.ipynb, the corresponding package is required. If not yet installed, run `pip install -U scikit-learn` in terminal.


## How-to-use
The interface is very similar to that of scikit-learn. 
```python
class randomforest.RandomForest(feature_count=1, tree_num=10,
                 depth=10, min_improv=1e-8, eval_func="gini_impurity")
```

### Tunable parameters include:
- `feature_count: integer, optional (default=1)`, number of subset features selected.
- `tree_num: integer, optional (default=10)`, number of decision trees to use in forest.
- `depth: integer, optional (default=10)`, maximum depth of each decision tree.
- `min_improv: float, optional (default=1e-8)`, minimum improvement in gini impurity/entropy required to split a node further.
- `eval_func: string, optional (default="gini_impurity")`, evaluation criteria, either gini impurity `"gini_impurity"` or entropy `"entropy"`.

### Methods:
- `fit(self, X, y)`, fits random forest of trees from training set `(X, y)`:
  - `X` is numpy array of shape [ n_samples x n_features ].
  - `y` is numpy array of shape [ n_samples ].
- `predict(self, X, rule="prob")`, predicts labels given dataset `X` and prediction rule.
  - `X` is numpy array of shape [ n_samples x n_features ]
  - `rule, optional (default="prob")` can also be `"majority"` whereby prediction is based on majority ruling from each decision trees.

## Sample Use
Run sample_test.ipynb for a quick demo and comparing the efficacy of random forest against other algoirthms.  
An example use code snippet is shown below.
```python
...
forest = RandomForest()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("Accuracy = %f" % sum(y_test != y_pred)/len(y_test))
```
