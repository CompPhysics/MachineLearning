import numpy as np

class LogisticRegression:
    """
    Logistic Regression for binary and multiclass classification.
    """
    def __init__(self, lr=0.01, epochs=1000, fit_intercept=True, verbose=False):
        self.lr = lr                  # Learning rate for gradient descent
        self.epochs = epochs          # Number of iterations
        self.fit_intercept = fit_intercept  # Whether to add intercept (bias)
        self.verbose = verbose        # Print loss during training if True
        self.weights = None
        self.multi_class = False      # Will be determined at fit time

    def _add_intercept(self, X):
        """Add intercept term (column of ones) to feature matrix."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z):
        """Sigmoid function for binary logistic."""
        return 1 / (1 + np.exp(-z))

    def _softmax(self, Z):
        """Softmax function for multiclass logistic."""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        Supports binary (sigmoid) and multiclass (softmax) based on y.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # Add intercept if needed
        if self.fit_intercept:
            X = self._add_intercept(X)
            n_features += 1

        # Determine classes and mode (binary vs multiclass)
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            self.multi_class = True
        else:
            self.multi_class = False

        # ----- Multiclass case -----
        if self.multi_class:
            n_classes = len(unique_classes)
            # Map original labels to 0...n_classes-1
            class_to_index = {c: idx for idx, c in enumerate(unique_classes)}
            y_indices = np.array([class_to_index[c] for c in y])
            # Initialize weight matrix (features x classes)
            self.weights = np.zeros((n_features, n_classes))

            # One-hot encode y
            Y_onehot = np.zeros((n_samples, n_classes))
            Y_onehot[np.arange(n_samples), y_indices] = 1

            # Gradient descent
            for epoch in range(self.epochs):
                scores = X.dot(self.weights)          # Linear scores (n_samples x n_classes)
                probs = self._softmax(scores)        # Probabilities (n_samples x n_classes)
                # Compute gradient (features x classes)
                gradient = (1 / n_samples) * X.T.dot(probs - Y_onehot)
                # Update weights
                self.weights -= self.lr * gradient

                if self.verbose and epoch % 100 == 0:
                    # Compute current loss (categorical cross-entropy)
                    loss = -np.sum(Y_onehot * np.log(probs + 1e-15)) / n_samples
                    print(f"[Epoch {epoch}] Multiclass loss: {loss:.4f}")

        # ----- Binary case -----
        else:
            # Convert y to 0/1 if not already
            if not np.array_equal(unique_classes, [0, 1]):
                # Map the two classes to 0 and 1
                class0, class1 = unique_classes
                y_binary = np.where(y == class1, 1, 0)
            else:
                y_binary = y.copy().astype(int)

            # Initialize weights vector (features,)
            self.weights = np.zeros(n_features)

            # Gradient descent
            for epoch in range(self.epochs):
                linear_model = X.dot(self.weights)     # (n_samples,)
                probs = self._sigmoid(linear_model)   # (n_samples,)
                # Gradient for binary cross-entropy
                gradient = (1 / n_samples) * X.T.dot(probs - y_binary)
                self.weights -= self.lr * gradient

                if self.verbose and epoch % 100 == 0:
                    # Compute binary cross-entropy loss
                    loss = -np.mean(
                        y_binary * np.log(probs + 1e-15) + 
                        (1 - y_binary) * np.log(1 - probs + 1e-15)
                    )
                    print(f"[Epoch {epoch}] Binary loss: {loss:.4f}")

    def predict_prob(self, X):
        """
        Compute probability estimates. Returns a 1D array for binary or
        a 2D array (n_samples x n_classes) for multiclass.
        """
        X = np.array(X)
        # Add intercept if the model used it
        if self.fit_intercept:
            X = self._add_intercept(X)
        scores = X.dot(self.weights)
        if self.multi_class:
            return self._softmax(scores)
        else:
            return self._sigmoid(scores)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        Returns integer class labels (0,1 for binary, or 0...C-1 for multiclass).
        """
        probs = self.predict_prob(X)
        if self.multi_class:
            # Choose class with highest probability
            return np.argmax(probs, axis=1)
        else:
            # Threshold at 0.5 for binary
            return (probs >= 0.5).astype(int)

"""
The class implements the sigmoid and softmax internally. During fit(), we check the number of classes: if more than 2, we set self.multi_class=True and perform multinomial logistic regression. We one-hot encode the target vector and update a weight matrix with softmax probabilities. Otherwise, we do standard binary logistic regression, converting labels to 0/1 if needed and updating a weight vector. In both cases we use batch gradient descent on the cross-entropy loss (we add a small epsilon 1e-15 to logs for numerical stability). Progress (loss) can be printed if verbose=True.
"""

# Evaluation Metrics
#We define helper functions for accuracy and cross-entropy loss. Accuracy is the fraction of correct predictions . For loss, we compute the appropriate cross-entropy:

def accuracy_score(y_true, y_pred):
    """Accuracy = (# correct predictions) / (total samples)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def binary_cross_entropy(y_true, y_prob):
    """
    Binary cross-entropy loss.
    y_true: true binary labels (0 or 1), y_prob: predicted probabilities for class 1.
    """
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-15, 1-1e-15)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def categorical_cross_entropy(y_true, y_prob):
    """
    Categorical cross-entropy loss for multiclass.
    y_true: true labels (0...C-1), y_prob: array of predicted probabilities (n_samples x C).
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.clip(np.array(y_prob), 1e-15, 1-1e-15)
    # One-hot encode true labels
    n_samples, n_classes = y_prob.shape
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(n_samples), y_true] = 1
    # Compute cross-entropy
    loss_vec = -np.sum(one_hot * np.log(y_prob), axis=1)
    return np.mean(loss_vec)


"""
Synthetic data generation
Binary classification data: Create two Gaussian clusters in 2D. For example, class 0 around mean [-2,-2] and class 1 around [2,2].
Multiclass data: Create several Gaussian clusters (one per class) spread out in feature space.
"""


import numpy as np

def generate_binary_data(n_samples=100, n_features=2, random_state=None):
    """
    Generate synthetic binary classification data.
    Returns (X, y) where X is (n_samples x n_features), y in {0,1}.
    """
    rng = np.random.RandomState(random_state)
    # Half samples for class 0, half for class 1
    n0 = n_samples // 2
    n1 = n_samples - n0
    # Class 0 around mean -2, class 1 around +2
    mean0 = -2 * np.ones(n_features)
    mean1 =  2 * np.ones(n_features)
    X0 = rng.randn(n0, n_features) + mean0
    X1 = rng.randn(n1, n_features) + mean1
    X = np.vstack((X0, X1))
    y = np.array([0]*n0 + [1]*n1)
    return X, y

def generate_multiclass_data(n_samples=150, n_features=2, n_classes=3, random_state=None):
    """
    Generate synthetic multiclass data with n_classes Gaussian clusters.
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    for cls in range(n_classes):
        # Random cluster center for each class
        center = rng.uniform(-5, 5, size=n_features)
        Xi = rng.randn(samples_per_class, n_features) + center
        yi = [cls] * samples_per_class
        X.append(Xi)
        y.extend(yi)
    X = np.vstack(X)
    y = np.array(y)
    return X, y


# Generate and test on binary data
X_bin, y_bin = generate_binary_data(n_samples=200, n_features=2, random_state=42)
model_bin = LogisticRegression(lr=0.1, epochs=1000)
model_bin.fit(X_bin, y_bin)
y_prob_bin = model_bin.predict_prob(X_bin)      # probabilities for class 1
y_pred_bin = model_bin.predict(X_bin)           # predicted classes 0 or 1

acc_bin = accuracy_score(y_bin, y_pred_bin)
loss_bin = binary_cross_entropy(y_bin, y_prob_bin)
print(f"Binary Classification - Accuracy: {acc_bin:.2f}, Cross-Entropy Loss: {loss_bin:.2f}")
#For multiclass:
# Generate and test on multiclass data
X_multi, y_multi = generate_multiclass_data(n_samples=300, n_features=2, n_classes=3, random_state=1)
model_multi = LogisticRegression(lr=0.1, epochs=1000)
model_multi.fit(X_multi, y_multi)
y_prob_multi = model_multi.predict_prob(X_multi)     # (n_samples x 3) probabilities
y_pred_multi = model_multi.predict(X_multi)          # predicted labels 0,1,2

acc_multi = accuracy_score(y_multi, y_pred_multi)
loss_multi = categorical_cross_entropy(y_multi, y_prob_multi)
print(f"Multiclass Classification - Accuracy: {acc_multi:.2f}, Cross-Entropy Loss: {loss_multi:.2f}")

# CSV Export
import csv

# Export binary results
with open('binary_results.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["TrueLabel", "PredictedLabel"])
    for true, pred in zip(y_bin, y_pred_bin):
        writer.writerow([true, pred])

# Export multiclass results
with open('multiclass_results.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["TrueLabel", "PredictedLabel"])
    for true, pred in zip(y_multi, y_pred_multi):
        writer.writerow([true, pred])

