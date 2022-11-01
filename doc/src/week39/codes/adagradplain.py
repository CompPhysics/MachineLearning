# Using Autograd to calculate gradients using AdaGrad and Stochastic Gradient descent
# OLS example
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# Note change from previous example
def CostOLS(theta):
    return (1.0/n)*np.sum((y-X @ theta)**2)

n = 1000
x = np.random.rand(n,1)
y = 2.0+3*x# +4*x*x

X = np.c_[np.ones((n,1)), x]
XT_X = X.T @ X
theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)


# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS)
theta = np.random.randn(2,1)
iterations = 1000
# Value for learning rate
eta = 0.01
# Including AdaGrad parameter to avoid possible division by zero
delta  = 1e-8
Giter = 0.0
for iter in range(iterations):
    gradients = training_gradient(theta)
    Giter += gradients*gradients
    update = gradients*eta/(delta+np.sqrt(Giter))
    theta -= update
print("theta from own AdaGrad")
print(theta)

