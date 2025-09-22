import numpy as np
import pandas as pd

# Load the dataset (ensure the CSV file is in the working directory)
df = pd.read_csv('creditcard.csv')

# Preprocess the data
df = df.drop('ID', axis=1)  # drop ID column
# Rename the target column for convenience (optional)
df = df.rename(columns={'default payment next month': 'default'})
# Separate features and target
X = df.drop('default', axis=1).values  # features (shape: [30000, 23])
y = df['default'].values              # target (shape: [30000,])

# Standardize features (zero mean, unit std dev)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Split into training and test sets (80/20 split)
m = X.shape[0]
train_size = int(0.8 * m)
indices = np.random.permutation(m)  # shuffle indices for randomness
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Add intercept term (bias) to features by adding a column of 1s
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize logistic regression parameters
n_features = X_train.shape[1]  # number of features including bias
theta = np.zeros(n_features)   # model weights (initialized to 0)

# Set training hyperparameters
learning_rate = 0.1
epochs = 500

# Training loop (gradient descent)
m_train = X_train.shape[0]
for epoch in range(epochs):
    # Compute predictions (sigmoid of linear combination)
    z = X_train.dot(theta)                   # linear combination
    predictions = 1 / (1 + np.exp(-z))       # sigmoid function
    
    # Compute the gradient of loss w.r.t. theta
    error = predictions - y_train            # vector of (pred - true) for each example
    grad = (X_train.T.dot(error)) / m_train  # gradient vector
    
    # Update weights
    theta -= learning_rate * grad
    
    # (Optional) compute and print loss every 50 epochs for monitoring
    if epoch % 50 == 0:
        # Binary cross-entropy loss
        loss = -np.mean(y_train * np.log(predictions + 1e-15) + 
                        (1 - y_train) * np.log(1 - predictions + 1e-15))
        print(f"Epoch {epoch}: Training loss = {loss:.4f}")

# 5. Evaluate the model on training and test data
# Predict probabilities for test set and classify as 1 if sigmoid >= 0.5
# (We can equivalently check linear term >= 0, since sigma(x)>=0.5 iff x>=0)
train_prob = 1 / (1 + np.exp(-X_train.dot(theta)))
test_prob  = 1 / (1 + np.exp(-X_test.dot(theta)))
train_pred = (train_prob >= 0.5).astype(int)
test_pred  = (test_prob  >= 0.5).astype(int)

# Calculate accuracy
train_accuracy = (train_pred == y_train).mean()
test_accuracy = (test_pred == y_test).mean()

# Calculate final loss on training set for reference
final_loss = -np.mean(y_train * np.log(train_prob + 1e-15) + 
                      (1 - y_train) * np.log(1 - train_prob + 1e-15))

# Print performance metrics
print(f"Final Training Loss: {final_loss:.4f}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
