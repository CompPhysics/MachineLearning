#The covariance matrix and its eigenvalues the hard way
from random import random, seed
import numpy as np

def covariance(x, y, n):
    sum = 0.0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    for i in range(0, n):
        sum += (x[(i)]-mean_x)*(y[i]-mean_y)
    return  sum/n

n = 10

x = np.random.normal(size=n)
y = np.random.normal(size=n)
z = x*x*x+y*y +0.5*np.random.normal(size=n)
covxx = covariance(x,x,n)
covxy = covariance(x,y,n)
covxz = covariance(x,z,n)
covyy = covariance(y,y,n)
covyz = covariance(y,z,n)
covzz = covariance(z,z,n)

SigmaCov = np.array([ [covxx, covxy, covxz], [covxy, covyy, covyz], [covxz, covyz, covzz]])
print(SigmaCov)

EigValues, EigVectors = np.linalg.eig(SigmaCov)
# sort eigenvectors and eigenvalues
permute = EigValues.argsort()
EigValues = EigValues[permute]
EigVectors = EigVectors[:,permute]
print(EigValues)
print(EigVectors)
