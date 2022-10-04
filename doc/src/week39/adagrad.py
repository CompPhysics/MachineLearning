# Using Autograd to calculate gradients using AdaGrad and Stochastic Gradient descent
# OLS example
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt

n = 10000
x = np.random.rand(n,1)
y = 4*x+3*x*x
# Setting up Design matrix
X = np.c_[np.ones((n,1)), x, x*x]
XTX = X.T @ X
XTy = X.T @ y
theta_linreg = np.linalg.pinv(XTX) @ (XTy)
print("Own inversion")
print(theta_linreg)


beta = np.random.randn(3,1)
eta = 0.01
delta = 1e-8
Niterations = 10000
Giter = np.zeros(shape=(3,3))
for iter in range(Niterations):
    gradient = (2.0/n)*(XTX @ beta - XTy)
    Giter +=gradient @ gradient.T
    Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
    beta -= np.multiply(Ginverse,gradient)

print("Optimal parameters with AdaGrad",beta)
