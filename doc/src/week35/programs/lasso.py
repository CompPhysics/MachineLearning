import numpy as np

# Seed for reproducibility
np.random.seed(0)

# Dimensions of the synthetic dataset
N = 100   # number of samples (observations)
p = 10    # number of features

# True sparse coefficients (only a few non-zero)
w_true = np.array([5, -3, 0, 0, 2, 0, 0, 0, 0, 0], dtype=float)
# For example, feature 0 has coefficient 5, feature 1 has -3, feature 4 has 2, rest are 0.

# Generate feature matrix X from a normal distribution
X = np.random.randn(N, p)

# Generate target values: linear combination of X with w_true + noise
noise = np.random.randn(N) * 1.0   # noise with standard deviation 1.0
y = X.dot(w_true) + noise


# Standardize features (zero mean, unit variance for each column)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0             # avoid division by zero if any constant feature
X_norm = (X - X_mean) / X_std

# Center the target to zero mean
y_mean = y.mean()
y_centered = y - y_mean


def soft_threshold(rho, lam):
    """Soft thresholding operator: S(rho, lam) = sign(rho)*max(|rho|-lam, 0)."""
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def lasso_coordinate_descent(X, y, alpha, max_iter=1000, tol=1e-6):
    """
    Perform LASSO regression using coordinate descent.
    X : array of shape (n_samples, n_features), assumed to be standardized.
    y : array of shape (n_samples,), assumed centered.
    alpha : regularization strength (L1 penalty coefficient).
    max_iter : maximum number of coordinate descent iterations (full cycles).
    tol : tolerance for convergence (stop if max coef change < tol).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # initialize weights to zero
    for it in range(max_iter):
        w_old = w.copy()
        # Loop over each feature coordinate
        for j in range(n_features):
            # Compute rho_j = x_j^T (y - X w + w_j * x_j)
            # (This is the contribution of feature j to the residual)
            X_j = X[:, j]
            # temporarily exclude feature j's effect
            residual = y - X.dot(w) + w[j] * X_j  
            rho_j = X_j.dot(residual)
            # Soft thresholding update for w_j
            w[j] = soft_threshold(rho_j, alpha) / (X_j.dot(X_j))
            # Check convergence: if all updates are very small, break
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


alpha = 50.0  # regularization strength
w_learned = lasso_coordinate_descent(X_norm, y_centered, alpha)

print("True coefficients:", w_true)
print("Learned coefficients:", w_learned)


# Plot y vs a relevant feature (0) and an irrelevant feature (2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X[:, 0], y, color='blue', alpha=0.6)
axes[0].set_title("Feature 0 (Relevant) vs Target")
axes[0].set_xlabel("Feature 0 values")
axes[0].set_ylabel("Target (y)")
axes[1].scatter(X[:, 2], y, color='red', alpha=0.6)
axes[1].set_title("Feature 2 (Irrelevant) vs Target")
axes[1].set_xlabel("Feature 2 values")
axes[1].set_ylabel("Target (y)")
plt.tight_layout()
plt.show()

# Track cost history during coordinate descent for plotting
def lasso_with_cost_history(X, y, alpha, max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    cost_history = []
    # initial cost
    cost_history.append(0.5 * np.sum((y - X.dot(w))**2) + alpha * np.sum(np.abs(w)))
    for it in range(max_iter):
        w_old = w.copy()
        for j in range(n_features):
            X_j = X[:, j]
            residual = y - X.dot(w) + w[j] * X_j
            rho_j = X_j.dot(residual)
            w[j] = soft_threshold(rho_j, alpha) / (X_j.dot(X_j))
        # compute cost after this iteration
        cost = 0.5 * np.sum((y - X.dot(w))**2) + alpha * np.sum(np.abs(w))
        cost_history.append(cost)
        if np.max(np.abs(w - w_old)) < 1e-6:
            break
    return w, cost_history

# Run coordinate descent and get cost history
w_fit, cost_history = lasso_with_cost_history(X_norm, y_centered, alpha=50.0)

# Plot cost vs iteration
plt.figure(figsize=(6,4))
plt.plot(cost_history, marker='o', color='purple')
plt.title("LASSO Cost Decrease over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost function value")
plt.grid(True)
plt.show()


# Compare true vs learned coefficients
import numpy as np
import matplotlib.pyplot as plt

indices = np.arange(p)
width = 0.4
plt.figure(figsize=(6,4))
plt.bar(indices - width/2, w_true, width=width, label='True Coefficient')
plt.bar(indices + width/2, w_fit, width=width, label='Learned Coefficient')
plt.xlabel("Feature index")
plt.ylabel("Coefficient value")
plt.title("True vs Learned Coefficients")
plt.legend()
plt.show()

