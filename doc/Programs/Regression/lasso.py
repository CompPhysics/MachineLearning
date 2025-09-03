import numpy as np
import matplotlib.pyplot as plt

class LassoGD:
    def __init__(self, lr=0.01, l1_penalty=1.0, tol=1e-6, max_iter=1000):
        """Initialize LASSO regressor with given hyperparameters."""
        self.lr = lr                  # learning rate (step size)
        self.l1_penalty = l1_penalty  # L1 regularization strength (λ)
        self.tol = tol                # convergence tolerance for change in cost
        self.max_iter = max_iter      # maximum iterations to run
        self.weights = None           # model weights (including intercept as w[0])
        self.feature_means = None     # to store feature means for normalization
        self.feature_stds = None      # to store feature std devs for normalization
        self.cost_history = None      # to record cost at each iteration

    def fit(self, X, y, normalize=True):
        """Train the LASSO model on data X (shape m x n) and targets y (length m)."""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        m, n = X.shape
        # 1. Feature normalization (zero mean, unit variance)
        if normalize:
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0  # avoid division by zero
            X = (X - self.feature_means) / self.feature_stds
        else:
            # If not normalizing, set means=0 and stds=1 for consistency
            self.feature_means = np.zeros(n)
            self.feature_stds = np.ones(n)
        # Add bias term (intercept) as an extra column of ones in X
        X_bias = np.hstack([np.ones((m, 1)), X])
        # Initialize weights (n features + 1 intercept) to zero
        self.weights = np.zeros(n + 1)
        self.cost_history = []

        prev_cost = float('inf')
        # Gradient Descent Loop
        for it in range(self.max_iter):
            # 2. Predictions for current weights
            y_pred = X_bias.dot(self.weights)
            error = y_pred - y
            # 3. Compute cost = MSE + L1 penalty (do not penalize intercept w[0])
            mse_cost = (error ** 2).mean() / 2.0
            l1_cost = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
            cost = mse_cost + l1_cost
            self.cost_history.append(cost)
            # Check convergence: stop if change in cost is below tolerance
            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost
            # 4. Compute gradient of MSE part
            grad_mse = (X_bias.T.dot(error)) / m  # gradient of 1/(2m)*RSS is X^T(error)/m
            # 5. Perform gradient descent update with L1 penalty via soft-thresholding
            # Take a gradient step for MSE
            w_temp = self.weights - self.lr * grad_mse
            # Soft-thresholding for L1: shrink weights toward 0 by lr*λ
            thresh = self.lr * self.l1_penalty
            w0 = w_temp[0]  # intercept (no regularization)
            w_rest = w_temp[1:]
            # Apply soft threshold to each weight in w_rest
            w_rest_updated = np.sign(w_rest) * np.maximum(np.abs(w_rest) - thresh, 0.0)
            # Update weights (combine intercept and rest)
            self.weights = np.concatenate(([w0], w_rest_updated))
        # End of gradient descent loop

    def predict(self, X):
        """Make predictions using the trained model on new data X."""
        X = np.array(X, dtype=float)
        # Normalize using the training mean and std
        X_norm = (X - self.feature_means) / self.feature_stds
        # Add bias term
        X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
        return X_bias.dot(self.weights)

# --- Example usage on a synthetic dataset ---
np.random.seed(0)
# Create synthetic data: m samples, n features
m, n = 100, 5
X = np.random.randn(m, n)
# True underlying weights for features (some are zero to illustrate feature selection)
true_w = np.array([0, 0, 5, 0, -3], dtype=float)
true_intercept = 10.0
# Generate targets with a linear combination of X and noise
y = true_intercept + X.dot(true_w) + 0.5 * np.random.randn(m)

# Train LASSO regression model
model = LassoGD(lr=0.05, l1_penalty=0.5, tol=1e-6, max_iter=1000)
model.fit(X, y)
print("Learned weights (intercept + coefficients):", model.weights)

# Plot the cost function history over iterations
plt.figure(figsize=(6,4))
plt.plot(model.cost_history, label="Cost")
plt.title("Cost Function Value vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE + L1 penalty)")
plt.legend()
plt.grid(True)
plt.show()
