# Importing various packages
from random import random, seed
import numpy as np

# the number of datapoints with a 2nd-order polynomial
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+5*x*x
# Design matrix
X = np.c_[np.ones((n,1)), x, x*x]
# Learning rate and number of iterations
eta = 0.05
Niterations = 100

# OLS part
beta_OLS = np.random.randn(3,1)
gradient = np.zeros(3)
for iter in range(Niterations):
    gradient = (2.0/n)*X.T @ (X @ beta_OLS-y)
    beta_OLS -= eta*gradient
print('Parameters for OLS using gradient descent')    
print(beta_OLS)

#Ridge and Lasso parameter Lambda
Lambda  = 0.01
Id = n*Lambda* np.eye((X.T @ X).shape[0])
# Gradient descent with  Ridge
beta_Ridge = np.random.randn(3,1)
gradient = np.zeros(3)
for iter in range(Niterations):
    gradients = 2.0/n*X.T @ (X @ beta_Ridge-y)+2*Lambda*beta_Ridge
    beta_Ridge -= eta*gradients
print('Parameters for Ridge using gradient descent')    
print(beta_Ridge)

# Gradient descent with Lasso
beta_Lasso = np.random.randn(3,1)
gradient = np.zeros(3)
for iter in range(Niterations):
    gradients = 2.0/n*X.T @ (X @ beta_Lasso-y)+Lambda*np.sign(beta_Lasso)
    beta_Lasso -= eta*gradients
print('Parameters for Lasso using gradient descent')    
print(beta_Lasso)
