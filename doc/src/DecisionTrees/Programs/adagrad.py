import numpy as np
class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gradient_squared = None
    def update(self, weights, gradient):
        if self.gradient_squared is None:
            self.gradient_squared = np.zeros_like(weights)
        # Accumulate squared gradients
        self.gradient_squared += gradient ** 2
        
        # Update weights
        adjusted_grads = gradient / (np.sqrt(self.gradient_squared) + self.epsilon)
        weights -= self.learning_rate * adjusted_grads
        
        return weights
# Example usage
if __name__ == "__main__":
    # Sample data and gradient
    np.random.seed(0)
    weights = np.random.rand(3)
    gradients = np.random.rand(100, 3)  # Simulating 100 gradients
    optimizer = AdaGrad(learning_rate=0.1)
    for grad in gradients:
        weights = optimizer.update(weights, grad)
    print("Updated weights:", weights)
