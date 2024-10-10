import numpy as np
class LassoRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights)
            gradient = (1 / num_samples) * np.dot(X.T, (linear_model - y)) + self.lambda_reg * np.sign(self.weights)
            # Update weights
            self.weights -= self.learning_rate * gradient
    def predict(self, X):
        return np.dot(X, self.weights)
# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])
    model = LassoRegression(learning_rate=0.01, num_iterations=1000, lambda_reg=0.1)
    model.fit(X, y)
    predictions = model.predict(X)
    print("Predictions:", predictions)
