# Using Autograd to calculate gradients using Adam and Stochastic Gradient descent
# OLS example
from random import random, seed
import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

# Note change from previous example
def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

n = 10000
x = np.random.rand(n,1)
y = 2.0+3*x +4*x*x# +np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x, x*x]
XT_X = X.T @ X
theta_linreg = np.linalg.pinv(XT_X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)


# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS,2)
# Define parameters for Stochastic Gradient Descent
n_epochs = 50
M = 5   #size of each minibatch
m = int(n/M) #number of minibatches
# Guess for unknown parameters theta
theta = np.random.randn(3,1)

# Value for learning rate
eta = 0.01
rho1 = 0.9
rho2 = 0.99
# Including AdaGrad parameter to avoid possible division by zero
delta  = 1e-8
for epoch in range(n_epochs):
    Giter = np.zeros(shape=(3,3))
    t = 0
    s = np.zeros(shape=(3,1))
    for i in range(m):
        random_index = M*np.random.randint(m)
        xi = X[random_index:random_index+M]
        yi = y[random_index:random_index+M]
        gradients = (1.0/M)*training_gradient(yi, xi, theta)
        t += 1
        Previous = Giter
        Giter +=gradients @ gradients.T
        Gnew = (rho2*Previous+(1-rho2)*Giter)
        Gnew = Gnew#/(1.0-rho2*t)
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
        s += rho1*s+(1-rho1)*gradients
#        snew = s/(1.0-rho1*t)
        theta -= Ginverse.T @ s
print("theta from own Adam")
print(theta)


from math import sqrt
