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
nsamples = 1000
x = np.random.randn(nsamples)
y = 3*x**2 + np.random.randn(nsamples)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)


# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE using OLS
scores_KFoldTrain = np.zeros((k))
scores_KFoldTest = np.zeros((k))
model = LinearRegression() 
j = 0
for train_inds, test_inds in kfold.split(x):
    xtrain = x[train_inds]
    ytrain = y[train_inds]
    xtest = x[test_inds]
    ytest = y[test_inds]
    Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
    model.fit(Xtrain, ytrain[:, np.newaxis])
    ypredtrain = model.predict(Xtrain)
    scores_KFoldTrain[j] = np.sum((ypredtrain - ytrain[:, np.newaxis])**2)/np.size(ypredtrain)
    print(f"Score for each fold train data:{scores_KFoldTrain[j]}")
    Xtest = poly.fit_transform(xtest[:, np.newaxis])
    ypredtest = model.predict(Xtest)
    scores_KFoldTest[j] = np.sum((ypredtest - ytest[:, np.newaxis])**2)/np.size(ypredtest)
    print(f"Score for each fold test data:{scores_KFoldTest[j]}")
    j += 1


estimated_mse_KFoldTest = np.mean(scores_KFoldTest)
print(f"Average OLS score for test:{estimated_mse_KFoldTest}")

estimated_mse_KFoldTrain = np.mean(scores_KFoldTrain)
print(f"Average OLS score for Train:{estimated_mse_KFoldTrain}")

