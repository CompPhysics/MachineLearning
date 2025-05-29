import numpy as np

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, max_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # Leaf class label

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape
        # Stop if conditions met
        if num_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1:
            leaf_val = self._majority_class(y)
            return DecisionTreeClassifier.Node(value=leaf_val)
        # Find best split
        feat_idx, thr = self._best_split(X, y)
        if feat_idx is None:
            leaf_val = self._majority_class(y)
            return DecisionTreeClassifier.Node(value=leaf_val)
        # Split data
        left_mask = X[:, feat_idx] <= thr
        left_node = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        return DecisionTreeClassifier.Node(feature=feat_idx, threshold=thr, left=left_node, right=right_node)

    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_thr = None, None
        if self.criterion == 'gini':
            base_impurity = self._gini(y)
        else:
            base_impurity = self._entropy(y)
        n_features = X.shape[1]
        features = range(n_features)
        # Possibly sample subset of features (for Random Forest use)
        if self.max_features is not None:
            if isinstance(self.max_features, int) and self.max_features < n_features:
                features = np.random.choice(n_features, self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                k = int(n_features * self.max_features)
                features = np.random.choice(n_features, k, replace=False)
        for feat in features:
            X_col = X[:, feat]
            unique_vals = np.unique(X_col)
            if len(unique_vals) <= 1:
                continue
            # Try midpoints between sorted unique values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thr in thresholds:
                left_mask = X_col <= thr
                y_left, y_right = y[left_mask], y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # Compute impurity of the split
                if self.criterion == 'gini':
                    imp_left = self._gini(y_left)
                    imp_right = self._gini(y_right)
                else:
                    imp_left = self._entropy(y_left)
                    imp_right = self._entropy(y_right)
                p = float(len(y_left)) / len(y)
                gain = base_impurity - (p * imp_left + (1 - p) * imp_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p**2)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _majority_class(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_input(x, self.tree) for x in X])

    def _predict_input(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_input(x, node.left)
        else:
            return self._predict_input(x, node.right)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # Leaf output value

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape
        if num_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth) or np.var(y) == 0:
            leaf_val = np.mean(y)
            return DecisionTreeRegressor.Node(value=leaf_val)
        best_sse = float('inf')
        best_feat, best_thr = None, None
        n_features = X.shape[1]
        features = range(n_features)
        if self.max_features is not None:
            if isinstance(self.max_features, int) and self.max_features < n_features:
                features = np.random.choice(n_features, self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                k = int(n_features * self.max_features)
                features = np.random.choice(n_features, k, replace=False)
        for feat in features:
            X_col = X[:, feat]
            unique_vals = np.unique(X_col)
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
            for thr in thresholds:
                left_mask = X_col <= thr
                y_left, y_right = y[left_mask], y[~left_mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # Compute sum of squared errors (SSE)
                left_mean, right_mean = np.mean(y_left), np.mean(y_right)
                sse_left = np.sum((y_left - left_mean) ** 2)
                sse_right = np.sum((y_right - right_mean) ** 2)
                sse = sse_left + sse_right
                if sse < best_sse:
                    best_sse, best_feat, best_thr = sse, feat, thr
        if best_feat is None:
            leaf_val = np.mean(y)
            return DecisionTreeRegressor.Node(value=leaf_val)
        left_mask = X[:, best_feat] <= best_thr
        left_node = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        return DecisionTreeRegressor.Node(feature=best_feat, threshold=best_thr, left=left_node, right=right_node)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_input(x, self.tree) for x in X])

    def _predict_input(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_input(x, node.left)
        else:
            return self._predict_input(x, node.right)

# Random Forests (Classification and Regression)

"""
Random forests train an ensemble of decision trees on bootstrapped data subsets and average their outputs . For classification, the final class is the majority vote of all trees; for regression, the output is the average prediction . Key points:

Bootstrap sampling: Each tree is trained on a random sample (with replacement) of the data.
Feature randomness: When splitting, each node may consider only a random subset of features (parameter max_features).
Aggregation: Classification uses mode of tree predictions; regression uses mean. This reduces overfitting compared to a single tree .
"""

import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n_samples, n_features = X.shape
        # Determine how many features to try at each split
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_feats = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        elif isinstance(self.max_features, float):
            max_feats = int(n_features * self.max_features)
        else:
            max_feats = n_features
        # Build trees
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTreeClassifier(criterion='gini', 
                                          max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          max_features=max_feats)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T  # shape (n_samples, n_trees)
        y_pred = []
        for preds in tree_preds:
            vote = Counter(preds).most_common(1)[0][0]
            y_pred.append(vote)
        return np.array(y_pred)

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_feats = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        elif isinstance(self.max_features, float):
            max_feats = int(n_features * self.max_features)
        else:
            max_feats = n_features
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         max_features=max_feats)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        # Average predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

# Gradient Boosting (Classification and Regression)

"""
Gradient boosting builds an additive ensemble of trees by fitting each new tree on the residuals (errors) of the existing model, effectively performing gradient descent on a loss function . At each stage m:

Compute the pseudo-residuals r_{im} = -\partial L(y_i, F(x_i)) / \partial F(x_i) (the negative gradient) .
Fit a tree h_m(x) to these residuals.
Update the model: F_m(x) = F_{m-1}(x) + \gamma_m \, h_m(x) (with line-search multiplier \gamma_m or simply a learning rate).


For binary classification, we use the logistic loss: initialize F_0 = \log(p/(1-p)) (log-odds of positive class) and repeatedly fit trees to y - \sigma(F). For multiclass, we fit one-vs-rest models (one boosting ensemble per class) and predict the class with highest score. Key concepts:

Additive updates: Each tree’s prediction is scaled by a learning rate and added to the ensemble output.
Regression loss: Typically squared-error (L2) for regression, logistic (cross-entropy) for classification.
Weak learners: Trees are often shallow (small max_depth).
"""

import numpy as np
from math import log, exp

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.gammas = []
        self.initial_prediction = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y, dtype=float)
        # Initialize prediction with mean
        self.initial_prediction = np.mean(y)
        F = np.full_like(y, fill_value=self.initial_prediction, dtype=float)
        for _ in range(self.n_estimators):
            residual = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            pred = tree.predict(X)
            # Line search for optimal multiplier gamma
            gamma = np.dot(residual, pred) / (np.dot(pred, pred) + 1e-8)
            F = F + self.learning_rate * gamma * pred
            self.trees.append(tree)
            self.gammas.append(gamma)
        return self

    def predict(self, X):
        X = np.array(X)
        F = np.full(X.shape[0], fill_value=self.initial_prediction, dtype=float)
        for tree, gamma in zip(self.trees, self.gammas):
            F += self.learning_rate * gamma * tree.predict(X)
        return F

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []   # For multiclass, one model per class

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) <= 2:
            # Binary classification (labels may be 0/1 or not)
            # Map labels to 0/1
            if set(self.classes_) != {0, 1}:
                class0, class1 = self.classes_[0], self.classes_[1]
                y_bin = np.array([0 if yi==class0 else 1 for yi in y])
                self.class_map = {0: class0, 1: class1}
            else:
                y_bin = y
                self.class_map = None
            # Initialize log-odds
            p = np.clip(np.mean(y_bin), 1e-6, 1-1e-6)
            F = np.full(y_bin.shape, fill_value=log(p/(1-p)), dtype=float)
            self.initial_F = F[0]
            self.trees = []
            for _ in range(self.n_estimators):
                P = 1 / (1 + np.exp(-F))
                residual = y_bin - P
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, residual)
                pred = tree.predict(X)
                F = F + self.learning_rate * pred
                self.trees.append(tree)
        else:
            # Multiclass one-vs-rest
            for cls in self.classes_:
                y_binary = (y == cls).astype(int)
                model = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                   learning_rate=self.learning_rate,
                                                   max_depth=self.max_depth)
                model.fit(X, y_binary)
                self.models.append(model)
        return self

    def predict(self, X):
        X = np.array(X)
        if len(self.classes_) <= 2:
            # Binary
            F = np.full(X.shape[0], fill_value=self.initial_F, dtype=float)
            for tree in self.trees:
                F += self.learning_rate * tree.predict(X)
            P = 1 / (1 + np.exp(-F))
            y_pred = (P >= 0.5).astype(int)
            if self.class_map:
                inv_map = {v:k for k,v in self.class_map.items()}
                y_pred = np.array([inv_map[val] for val in y_pred])
            return y_pred
        else:
            # Multiclass: compute score for each class
            scores = []
            for model in self.models:
                F_cls = np.full(X.shape[0], fill_value=model.initial_F, dtype=float)
                for tree in model.trees:
                    F_cls += self.learning_rate * tree.predict(X)
                scores.append(F_cls)
            scores = np.vstack(scores).T  # shape (n_samples, n_classes)
            class_idx = np.argmax(scores, axis=1)
            return np.array([self.classes_[i] for i in class_idx])


# Classification Example: Generate a 2-class dataset, train models, and evaluate accuracy.
import numpy as np
import csv

# Synthetic binary classification data
np.random.seed(0)
N = 100
# Class 0 centered at (-2, -2), Class 1 at (2, 2)
X0 = np.random.randn(N, 2) - 2
X1 = np.random.randn(N, 2) + 2
X_clf = np.vstack([X0, X1])
y_clf = np.array([0]*N + [1]*N)

# Shuffle data
perm = np.random.permutation(len(y_clf))
X_clf, y_clf = X_clf[perm], y_clf[perm]

# Train models
dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(X_clf, y_clf)
rf_clf = RandomForestClassifier(n_estimators=10, max_depth=3)
rf_clf.fit(X_clf, y_clf)
gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
gb_clf.fit(X_clf, y_clf)

# Predictions
pred_dt = dt_clf.predict(X_clf)
pred_rf = rf_clf.predict(X_clf)
pred_gb = gb_clf.predict(X_clf)

# Accuracy evaluation
acc_dt = np.mean(pred_dt == y_clf)
acc_rf = np.mean(pred_rf == y_clf)
acc_gb = np.mean(pred_gb == y_clf)
print(f"Decision Tree Accuracy: {acc_dt:.2f}")
print(f"Random Forest Accuracy: {acc_rf:.2f}")
print(f"Gradient Boosting Accuracy: {acc_gb:.2f}")

# Export to CSV
with open('classification_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GroundTruth', 'DT_Pred', 'RF_Pred', 'GB_Pred'])
    for true, d, r, g in zip(y_clf, pred_dt, pred_rf, pred_gb):
        writer.writerow([true, d, r, g])

# Regression Example: Generate a simple regression dataset, train models, and compute MSE.
import numpy as np
import csv

# Synthetic regression data: y = 3*x1 - 2*x2 + noise
np.random.seed(1)
N = 200
X_reg = np.random.randn(N, 2)
y_reg = 3 * X_reg[:,0] - 2 * X_reg[:,1] + np.random.randn(N) * 0.5

# Train models
dt_reg = DecisionTreeRegressor(max_depth=4)
dt_reg.fit(X_reg, y_reg)
rf_reg = RandomForestRegressor(n_estimators=10, max_depth=4)
rf_reg.fit(X_reg, y_reg)
gb_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2)
gb_reg.fit(X_reg, y_reg)

# Predictions
pred_dt_r = dt_reg.predict(X_reg)
pred_rf_r = rf_reg.predict(X_reg)
pred_gb_r = gb_reg.predict(X_reg)

# MSE evaluation
mse_dt = np.mean((pred_dt_r - y_reg)**2)
mse_rf = np.mean((pred_rf_r - y_reg)**2)
mse_gb = np.mean((pred_gb_r - y_reg)**2)
print(f"Decision Tree MSE: {mse_dt:.3f}")
print(f"Random Forest MSE: {mse_rf:.3f}")
print(f"Gradient Boosting MSE: {mse_gb:.3f}")

# Export to CSV
with open('regression_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['GroundTruth', 'DT_Pred', 'RF_Pred', 'GB_Pred'])
    for true, d, r, g in zip(y_reg, pred_dt_r, pred_rf_r, pred_gb_r):
        writer.writerow([true, d, r, g])

