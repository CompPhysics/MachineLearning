
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionModel:
    """Linear Regression using ordinary least squares (L2 loss)."""
    def __init__(self):
        self.model = LinearRegression()  # scikit-learn linear regression

    def fit(self, X, y):
        """Fit the linear model to the data X (n_samples, n_features) and target y."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict target values for input X."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on data (X, y), returning Mean Squared Error and R².
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2
from sklearn.linear_model import Ridge

class RidgeRegressionModel:
    """Ridge Regression (Linear least squares with L2 regularization)."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2

from sklearn.linear_model import Lasso

class LassoRegressionModel:
    """Lasso Regression (Linear least squares with L1 regularization)."""
    def __init__(self, alpha=1.0):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2



from sklearn.kernel_ridge import KernelRidge

class KernelRidgeRegressionModel:
    """Kernel Ridge Regression (ridge regression in kernel space)."""
    def __init__(self, alpha=1.0, kernel='linear', gamma=None):
        """
        kernel: kernel type ('linear', 'rbf', 'poly', etc.)
        gamma: kernel coefficient for 'rbf', 'poly' etc. (if needed)
        """
        self.model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2


# Synthetic Data and Demonstration
# We now generate a synthetic regression dataset (with a single feature for easy plotting) and compare all models. We use 100 samples with some noise. Each model is trained on the same data, and we compute its MSE and R².

import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data: 100 samples, 1 feature, added noise
X, y = make_regression(n_samples=100, n_features=1, noise=20.0, random_state=42)
# Reshape X to be 2D (100,1)
X = X.reshape(-1, 1)

# Initialize models
lin_model = LinearRegressionModel()
ridge_model = RidgeRegressionModel(alpha=1.0)
lasso_model = LassoRegressionModel(alpha=1.0)
kernel_model = KernelRidgeRegressionModel(alpha=1.0, kernel='rbf', gamma=0.1)

# Fit models
lin_model.fit(X, y)
ridge_model.fit(X, y)
lasso_model.fit(X, y)
kernel_model.fit(X, y)

# Evaluate models
models = {
    "Linear": lin_model,
    "Ridge": ridge_model,
    "Lasso": lasso_model,
    "Kernel Ridge": kernel_model
}

for name, model in models.items():
    mse, r2 = model.evaluate(X, y)
    print(f"{name} Regression -> MSE: {mse:.2f}, R\u00b2: {r2:.3f}")

"""
In this demonstration, all four models are fit to the same synthetic data. We output each model’s MSE and R² (higher R² closer to 1 is better). Below is an example of plotting the linear model’s fit and its residuals. In a scatter plot (Figure 1) we overlay the fitted line from the linear model. A residual plot (Figure 2) shows the prediction errors versus predicted values, which helps diagnose any patterns or biases
"""

# Example plotting code (not required in the answer, just for illustration):
import matplotlib.pyplot as plt
y_pred_lin = lin_model.predict(X)
residuals = y - y_pred_lin

# Scatter and regression line
plt.scatter(X, y, color='blue', alpha=0.7, label='Data')
# Sort X for a smooth line
X_sorted = np.sort(X.flatten())
y_line = lin_model.predict(X_sorted.reshape(-1,1))
plt.plot(X_sorted, y_line, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Residual plot
plt.scatter(y_pred_lin, residuals, color='blue', alpha=0.7)
plt.hlines(0, xmin=y_pred_lin.min(), xmax=y_pred_lin.max(), colors='red', linestyles='--')
plt.xlabel('Predicted value')
plt.ylabel('Residual (True - Predicted)')
plt.title('Residuals vs Predicted (Linear Model)')
plt.show()
"""
The above object-oriented implementation and demo show how to train, predict, and evaluate different regression models in Python. Each model class is well-documented and easily reusable. The synthetic-data example compares models on the same data and visualizes both the fitted regression line and the residuals. This modular design makes it straightforward to extend or modify the models (for example, by tuning hyperparameters or adding cross-validation).
"""
