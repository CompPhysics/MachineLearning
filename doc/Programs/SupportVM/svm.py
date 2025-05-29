import numpy as np
import csv

class SVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000,
                 kernel='linear', degree=3, gamma=None, coef0=1, random_features=100):
        """
        SVM classifier with linear, polynomial, or (approximate) RBF kernel.
        Trains via stochastic subgradient descent on hinge loss.
        """
        self.C = C                  # Regularization parameter
        self.lr = lr                # Learning rate
        self.epochs = epochs        # Number of epochs for training
        self.kernel = kernel        # 'linear', 'poly', or 'rbf'
        self.degree = degree        # Degree for polynomial kernel
        self.coef0 = coef0          # Offset for polynomial kernel (not used in feature expansion here)
        self.gamma = gamma          # Kernel scale for RBF or polynomial (default 1/n_features)
        self.random_features = random_features  # Number of random Fourier features for RBF
        self.w = None               # Weight vector (in feature space)
        self.b = 0.0                # Bias term

        # For RBF approximation: random projection parameters
        self.W = None
        self.b_rff = None

    def _poly_features(self, X):
        """
        Explicit polynomial feature expansion up to self.degree.
        E.g., degree=2 for features [x1, x2] gives [x1, x2, x1^2, x1*x2, x2^2].
        """
        from itertools import combinations_with_replacement
        n_samples, n_features = X.shape
        features = []
        for deg in range(1, self.degree+1):
            for combo in combinations_with_replacement(range(n_features), deg):
                feat = np.ones(n_samples)
                for i in combo:
                    feat *= X[:, i]
                features.append(feat)
        if len(features) > 0:
            return np.vstack(features).T
        else:
            return np.zeros((n_samples, 0))

    def _rbf_features(self, X):
        """
        Random Fourier feature mapping for RBF kernel approximation:
        phi(x) = sqrt(2/D) * [cos(w_i^T x + b_i)]_i.
        """
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        D = self.random_features
        if self.W is None:
            self.W = np.sqrt(2 * self.gamma) * np.random.randn(D, X.shape[1])
        if self.b_rff is None:
            self.b_rff = 2 * np.pi * np.random.rand(D)
        # Compute RFF features
        return np.sqrt(2.0 / D) * np.cos(np.dot(X, self.W.T) + self.b_rff)

    def fit(self, X, y):
        """
        Train the SVM. y should be encoded as {-1, +1}.
        Uses subgradient descent on the hinge loss (primal form).
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # Feature mapping based on kernel choice
        if self.kernel == 'linear':
            X_train = X.copy()
        elif self.kernel == 'poly':
            X_train = self._poly_features(X)
        elif self.kernel == 'rbf':
            X_train = self._rbf_features(X)
        else:
            raise ValueError("Unsupported kernel type")

        n_samples, n_features = X_train.shape
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Ensure labels are -1 or +1
        y_signed = np.where(y <= 0, -1, 1)

        # Stochastic subgradient descent
        for epoch in range(self.epochs):
            for i in range(n_samples):
                xi, yi = X_train[i], y_signed[i]
                margin = yi * (np.dot(self.w, xi) + self.b)
                if margin < 1:
                    # Hinge loss is active: update gradient includes -C*y*x
                    grad_w = self.w - self.C * yi * xi
                    grad_b = -self.C * yi
                else:
                    # No loss: only regularizer gradient
                    grad_w = self.w
                    grad_b = 0.0
                # Update parameters
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def decision_function(self, X):
        """
        Compute the raw decision score (w·x + b) for samples X.
        """
        X = np.array(X, dtype=float)
        if self.kernel == 'linear':
            X_eval = X
        elif self.kernel == 'poly':
            X_eval = self._poly_features(X)
        elif self.kernel == 'rbf':
            X_eval = self._rbf_features(X)
        else:
            raise ValueError("Unsupported kernel type")
        return np.dot(X_eval, self.w) + self.b

    def predict(self, X):
        """
        Predict class labels {-1, +1} for samples in X.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def predict_proba(self, X):
        """
        (Approximate) probability estimates for binary classification.
        Uses a logistic function on the decision scores.
        Returns shape (n_samples, 2) with columns [P(y=-1), P(y=+1)].
        """
        scores = self.decision_function(X)
        # Sigmoid for positive class
        p_pos = 1.0 / (1.0 + np.exp(-scores))
        p_neg = 1.0 - p_pos
        return np.vstack([p_neg, p_pos]).T


class MultiClassSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000,
                 kernel='linear', degree=3, gamma=None, coef0=1):
        """
        One-vs-Rest multiclass SVM. Trains one binary SVM per class.
        """
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.classes_ = None
        self.models = {}

    def fit(self, X, y):
        """
        Fit one SVM for each class.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            # Prepare binary labels: +1 for current class, -1 for all others
            y_binary = np.where(y == cls, 1, -1)
            svm = SVM(C=self.C, lr=self.lr, epochs=self.epochs,
                      kernel=self.kernel, degree=self.degree,
                      gamma=self.gamma, coef0=self.coef0)
            svm.fit(X, y_binary)
            self.models[cls] = svm

    def predict(self, X):
        """
        Predict multiclass labels by picking the class with highest decision score.
        """
        X = np.array(X, dtype=float)
        # Compute decision score for each class model
        scores = np.vstack([self.models[cls].decision_function(X) for cls in self.classes_]).T
        # Choose class index with max score
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """
        Return (approximate) class probabilities via softmax on the decision scores.
        """
        X = np.array(X, dtype=float)
        scores = np.vstack([self.models[cls].decision_function(X) for cls in self.classes_]).T
        # Softmax for probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


class SVR:
    def __init__(self, C=1.0, lr=0.001, epochs=1000, epsilon=0.1,
                 kernel='linear', degree=3, gamma=None, coef0=1, random_features=100):
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_features = random_features
        self.w = None
        self.b = 0.0
        self.W = None
        self.b_rff = None

    def _poly_features(self, X):
        from itertools import combinations_with_replacement
        n_samples, n_features = X.shape
        features = []
        for deg in range(1, self.degree+1):
            for combo in combinations_with_replacement(range(n_features), deg):
                feat = np.ones(n_samples)
                for i in combo:
                    feat *= X[:, i]
                features.append(feat)
        if len(features) > 0:
            return np.vstack(features).T
        else:
            return np.zeros((n_samples, 0))

    def _rbf_features(self, X):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        D = self.random_features
        if self.W is None:
            self.W = np.sqrt(2 * self.gamma) * np.random.randn(D, X.shape[1])
        if self.b_rff is None:
            self.b_rff = 2 * np.pi * np.random.rand(D)
        return np.sqrt(2.0 / D) * np.cos(np.dot(X, self.W.T) + self.b_rff)

    def fit(self, X, y):
        """
        Train the SVR model. Uses subgradient descent on the ε-insensitive loss.
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        # Feature mapping for kernel
        if self.kernel == 'linear':
            X_train = X.copy()
        elif self.kernel == 'poly':
            X_train = self._poly_features(X)
        elif self.kernel == 'rbf':
            X_train = self._rbf_features(X)
        else:
            raise ValueError("Unsupported kernel type")
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Training loop
        for epoch in range(self.epochs):
            for i in range(n_samples):
                xi, yi = X_train[i], y[i]
                f = np.dot(self.w, xi) + self.b
                if yi - f > self.epsilon:
                    # Prediction too low: push up
                    grad_w = self.w - self.C * xi
                    grad_b = -self.C
                elif f - yi > self.epsilon:
                    # Prediction too high: push down
                    grad_w = self.w + self.C * xi
                    grad_b = self.C
                else:
                    # Within ε: only regularization gradient
                    grad_w = self.w
                    grad_b = 0.0
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def predict(self, X):
        """
        Predict continuous values for regression.
        """
        X = np.array(X, dtype=float)
        if self.kernel == 'linear':
            X_eval = X
        elif self.kernel == 'poly':
            X_eval = self._poly_features(X)
        elif self.kernel == 'rbf':
            X_eval = self._rbf_features(X)
        else:
            raise ValueError("Unsupported kernel type")
        return np.dot(X_eval, self.w) + self.b


# Evaluation metrics and utilities
def accuracy_score(y_true, y_pred):
    """Classification accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def mean_squared_error(y_true, y_pred):
    """Mean squared error for regression."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

def save_predictions_to_csv(y_true, y_pred, filename):
    """Save true vs. predicted values to a CSV file."""
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True', 'Predicted'])
        for true, pred in zip(y_true, y_pred):
            writer.writerow([true, pred])


# Synthetic data generation
def generate_classification_data(binary=True, n_samples=100, n_features=2, n_classes=2, random_state=None):
    """
    Generate synthetic classification data:
    - Binary: two Gaussian clusters.
    - Multiclass: 'n_classes' clusters with increasing means.
    """
    rng = np.random.RandomState(random_state)
    if binary and n_classes == 2:
        mean1 = rng.randn(n_features) - 1
        mean2 = rng.randn(n_features) + 1
        cov = np.eye(n_features) * 0.2
        X1 = rng.multivariate_normal(mean1, cov, size=n_samples//2)
        X2 = rng.multivariate_normal(mean2, cov, size=n_samples//2)
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    else:
        X = []
        y = []
        for cls in range(n_classes):
            center = rng.randn(n_features) + cls * 2.0
            cov = np.eye(n_features) * 0.3
            Xc = rng.multivariate_normal(center, cov, size=n_samples//n_classes)
            yc = np.full(n_samples//n_classes, cls)
            X.append(Xc);  y.append(yc)
        X = np.vstack(X)
        y = np.hstack(y)
    return X, y

def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=None):
    """
    Generate synthetic regression data: y = sin(x) + noise.
    """
    rng = np.random.RandomState(random_state)
    X = np.linspace(-5, 5, n_samples).reshape(-1, n_features)
    y = np.sin(X).ravel() + noise * rng.randn(n_samples)
    return X, y


# --- Binary Classification Demo ---
# Generate binary data
X_bin, y_bin = generate_classification_data(binary=True, n_samples=200, n_features=2, n_classes=2, random_state=1)
y_bin_signed = np.where(y_bin == 0, -1, 1)  # Encode labels as -1/+1

svm = SVM(C=1.0, lr=0.01, epochs=500, kernel='linear')
svm.fit(X_bin, y_bin_signed)
pred_bin = svm.predict(X_bin)
acc_bin = accuracy_score(y_bin_signed, pred_bin)
print("Binary SVM accuracy:", acc_bin)

# (Optional) Save predictions
save_predictions_to_csv(y_bin_signed, pred_bin, "binary_svm_results.csv")


# --- Multiclass Classification Demo ---
# Generate 3-class data
X_multi, y_multi = generate_classification_data(binary=False, n_samples=300, n_features=2, n_classes=3, random_state=42)

mc_svm = MultiClassSVM(C=1.0, lr=0.01, epochs=500, kernel='linear')
mc_svm.fit(X_multi, y_multi)
pred_multi = mc_svm.predict(X_multi)
acc_multi = accuracy_score(y_multi, pred_multi)
print("Multiclass SVM accuracy:", acc_multi)

# Show probability estimates for first few samples (softmaxed scores)
probs_multi = mc_svm.predict_proba(X_multi[:5])
print("Class probabilities (first 5):\n", probs_multi)

# --- Regression Demo ---
# Generate regression data
X_reg, y_reg = generate_regression_data(n_samples=100, noise=0.2, random_state=0)

# Linear SVR
svr = SVR(C=1.0, lr=0.01, epochs=1000, epsilon=0.1, kernel='linear')
svr.fit(X_reg, y_reg)
pred_reg = svr.predict(X_reg)
mse = mean_squared_error(y_reg, pred_reg)
print("SVR Mean Squared Error:", mse)

