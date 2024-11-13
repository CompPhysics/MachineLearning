import numpy as np

class DecisionStump:
    def fit(self, X, y, weights):
        m, n = X.shape
        self.alpha = 0
        self.threshold = None
        self.polarity = 1

        min_error = float('inf')

        for feature in range(n):
            feature_values = np.unique(X[:, feature])

            for threshold in feature_values:
                for polarity in [1, -1]:
                    predictions = np.ones(m)
                    predictions[X[:, feature] < threshold] = -1
                    predictions *= polarity

                    error = sum(weights[predictions != y])

                    if error < min_error:
                        min_error = error
                        self.alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
                        self.threshold = threshold
                        self.feature_index = feature
                        self.polarity = polarity

    def predict(self, X):
        m = X.shape[0]
        predictions = np.ones(m)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions

class AdaBoost:
    def fit(self, X, y, n_estimators):
        m = X.shape[0]
        self.alphas = []
        self.models = []

        weights = np.ones(m) / m

        for _ in range(n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            error = sum(weights[predictions != y])
            if error == 0:
                break

            self.models.append(stump)
            self.alphas.append(stump.alpha)

            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_predictions += alpha * model.predict(X)
        return np.sign(final_predictions)

# Example dataset (X, y)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1])  # Labels must be -1 or 1

# Train AdaBoost
ada = AdaBoost()
ada.fit(X, y, n_estimators=10)

# Predictions
predictions = ada.predict(X)
print("Predictions:", predictions)
