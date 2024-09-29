import numpy as np
class DecisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth < self.max_depth:
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is not None:
                left_indices = X[:, best_feature] < best_threshold
                right_indices = X[:, best_feature] >= best_threshold
                left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
                right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
                return (best_feature, best_threshold, left_child, right_child)
        return np.mean(y)
    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if len(y[left_indices]) > 0 and len(y[right_indices]) > 0:
                    left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                    right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)
                    mse = (len(y[left_indices]) * left_mse + len(y[right_indices]) * right_mse) / n_samples
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold
        return best_feature, best_threshold
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    def _predict_sample(self, sample, node):
        if isinstance(node, tuple):
            feature, threshold, left_child, right_child = node
            if sample[feature] < threshold:
                return self._predict_sample(sample, left_child)
            else:
                return self._predict_sample(sample, right_child)
        return node
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
    def fit(self, X, y):
        y_pred = np.zeros(y.shape)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            y_pred += self.learning_rate * model.predict(X)
            self.models.append(model)
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.5, 1.7, 3.5, 3.7, 5.0])
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
