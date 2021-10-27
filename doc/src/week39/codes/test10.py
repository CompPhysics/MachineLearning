"""
Code to test Ridge with own gradient descent and SGD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns
import autograd.numpy as np
from autograd import grad


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(315)

n = 100
x = np.random.rand(n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)

Maxpolydegree = 5
X = np.zeros((n,Maxpolydegree-1))

for degree in range(1,Maxpolydegree): #No intercept column
    X[:,degree-1] = x**(degree)

# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


nlambdas = 10
lmbd_vals = np.logspace(-4, 0, nlambdas)
MSERidgePredict = np.zeros(nlambdas)
for i in range(nlambdas):
    lmb = lmbd_vals[i]
    RegRidge = linear_model.Ridge(lmb,fit_intercept=False)
    RegRidge.fit(X_train,y_train)
    ypredictRidge = RegRidge.predict(X_test)
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)

beta = np.random.randn(X_train.shape[1],1)
loss = np.mean((y_train.reshape(-1,1) - X_train@beta)**2)
print(loss)
get_grad = grad(loss,argnum=2) 
grad_beta = get_grad(X_train,y_train,beta) 
