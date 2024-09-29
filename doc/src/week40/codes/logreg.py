import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights)
            y_predicted = self.sigmoid(linear_model)
            # Gradient calculation
            gradient = np.dot(X.T, (y_predicted - y)) / num_samples
            
            # Update weights
            self.weights -= self.learning_rate * gradient
    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_predicted]
# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
