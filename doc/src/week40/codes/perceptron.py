import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    def activation_function(self, x):
        return 1 if x >= 0 else 0
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = [self.activation_function(i) for i in linear_output]
        return np.array(y_predicted)
# Example usage
if __name__ == "__main__":
    # Sample data (AND logic gate)
    X = np.array([[0, 0], 
                  [0, 1], 
                  [1, 0], 
                  [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND outputs
    perceptron = Perceptron(learning_rate=0.1, n_iters=10)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    print("Final predictions:", predictions)
