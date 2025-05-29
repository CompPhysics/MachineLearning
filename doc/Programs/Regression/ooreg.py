import numpy as np
import csv
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_total

def save_csv(filename, X, y_true, y_pred):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "True Y", "Predicted Y"])
        for x, y, y_hat in zip(X, y_true, y_pred):
            writer.writerow([x[0], y, y_hat])

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.weights

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        n = X_bias.shape[1]
        I = np.eye(n)
        I[0, 0] = 0
        self.weights = np.linalg.inv(X_bias.T @ X_bias + self.alpha * I) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.weights

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        n_samples, n_features = X_bias.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            weights_old = self.weights.copy()
            for j in range(n_features):
                tmp = X_bias @ self.weights - X_bias[:, j] * self.weights[j]
                rho = np.dot(X_bias[:, j], y - tmp)
                if j == 0:
                    self.weights[j] = rho / np.sum(X_bias[:, j] ** 2)
                else:
                    if rho < -self.alpha / 2:
                        self.weights[j] = (rho + self.alpha / 2) / np.sum(X_bias[:, j] ** 2)
                    elif rho > self.alpha / 2:
                        self.weights[j] = (rho - self.alpha / 2) / np.sum(X_bias[:, j] ** 2)
                    else:
                        self.weights[j] = 0
            if np.linalg.norm(self.weights - weights_old, ord=1) < self.tol:
                break

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.weights

class KernelRidgeRegression:
    def __init__(self, alpha=1.0, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.X_train = None
        self.alpha_vec = None

    def _rbf_kernel(self, X1, X2):
        dists = np.sum((X1[:, np.newaxis] - X2[np.newaxis, :]) ** 2, axis=2)
        return np.exp(-self.gamma * dists)

    def fit(self, X, y):
        self.X_train = X
        K = self._rbf_kernel(X, X)
        n = K.shape[0]
        self.alpha_vec = np.linalg.inv(K + self.alpha * np.eye(n)) @ y

    def predict(self, X):
        K = self._rbf_kernel(X, self.X_train)
        return K @ self.alpha_vec

if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(100) * 0.5

    models = {
        "linear": LinearRegression(),
        "ridge": RidgeRegression(alpha=1.0),
        "lasso": LassoRegression(alpha=0.1),
        "kernel_ridge": KernelRidgeRegression(alpha=1.0, gamma=5.0)
    }

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mse_val = mse(y, y_pred)
        r2_val = r2_score(y, y_pred)
        print(f"{name} -> MSE: {mse_val:.4f}, R2: {r2_val:.4f}")
        save_csv(f"predictions_{name}.csv", X, y, y_pred)
