import numpy as np
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=float("inf")):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        # Check stopping criteria
        if (num_samples < self.min_samples_split or
                depth >= self.max_depth or
                len(unique_classes) == 1):
            return self.Node(value=self._most_common_label(y))
        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return self.Node(value=self._most_common_label(y))
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            num_samples = len(y)
            for i in range(1, num_samples):
                if classes[i] == classes[i - 1]:
                    continue
                
                threshold = (thresholds[i] + thresholds[i - 1]) / 2
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    def _information_gain(self, parent, left, right):
        total_samples = len(parent)
        if len(left) == 0 or len(right) == 0:
            return 0
        
        parent_entropy = self._entropy(parent)
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)
        weighted_entropy = (len(left) / total_samples) * left_entropy + (len(right) / total_samples) * right_entropy
        return parent_entropy - weighted_entropy
    def _entropy(self, y):
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        return -np.sum(probabilities * np.log(probabilities + 1e-10))
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])
    def _predict(self, inputs):
        node = self.tree
        while node.value is None:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value
# Example usage
if __name__ == "__main__":
    # Example dataset
    X = np.array([[2.5], [1.0], [1.5], [3.0], [3.5], [2.0], [4.0], [2.2]])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])  # Binary labels
    # Train decision tree
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    # Predictions
    predictions = tree.predict(X)
    print("Predictions:", predictions)        
