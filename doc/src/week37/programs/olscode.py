import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

# Generate the data.
nsamples = 10000
x = np.random.randn(nsamples)
y = 3*x**2 + np.random.randn(nsamples)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)


# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE using OLS
scores_KFold = np.zeros((k))
model = LinearRegression() 
j = 0
for train_inds, test_inds in kfold.split(x):
    xtrain = x[train_inds]
    ytrain = y[train_inds]
    xtest = x[test_inds]
    ytest = y[test_inds]
    Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
    model.fit(Xtrain, ytrain[:, np.newaxis])
    Xtest = poly.fit_transform(xtest[:, np.newaxis])
    ypred = model.predict(Xtest)
    scores_KFold[j] = np.sum((ypred - ytest[:, np.newaxis])**2)/np.size(ypred)
    print(f"Score for each fold:{scores_KFold[j]}")
    j += 1


estimated_mse_KFold = np.mean(scores_KFold)
print(f"Average OLS score:{estimated_mse_KFold}")

