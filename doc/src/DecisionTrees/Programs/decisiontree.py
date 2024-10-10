import numpy as np
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, output=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.output = output
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        # Check for stopping conditions
        if (num_samples < self.min_samples_split) or (depth == self.max_depth) or (len(unique_classes) == 1):
            output = self._most_common_label(y)
            return DecisionTreeNode(output=output)
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_features)
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionTreeNode(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            num_left = [0] * len(np.unique(y))
            num_right = [np.sum(classes == c) for c in np.unique(y)]
            for i in range(1, len(y)):  # At least one in each side
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gain = self._information_gain(num_left, num_right, len(classes), len(y))
                if thresholds[i] == thresholds[i - 1]:  # Skip duplicate values
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2  # Average threshold
        return best_feature, best_threshold
    def _information_gain(self, num_left, num_right, num_total, num_classes):
        p_left = float(len(num_left)) / num_total
        p_right = float(len(num_right)) / num_total
        entropy_before = self._entropy(num_left, num_total)
        entropy_left = self._entropy(num_left, sum(num_left))
        entropy_right = self._entropy(num_right, sum(num_right))
        entropy_after = p_left * entropy_left + p_right * entropy_right
        
        return entropy_before - entropy_after
    def _entropy(self, counts, total):
        if total == 0:
            return 0
        return -sum((count / total) * np.log2(count / total) for count in counts if count > 0)
    def _most_common_label(self, y):
        return np.bincount(y).argmax()
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])
    def _predict_sample(self, sample, node):
        if node.output is not None:
            return node.output
        if sample[node.feature_index] < node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
# Example usage
if __name__ == "__main__":
    # Sample data (AND gate)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])
    model = DecisionTree(min_samples_split=1, max_depth=3)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
