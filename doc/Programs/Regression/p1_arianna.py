#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:53:35 2019

@author: Ary
"""

import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

np.random.seed(2204)

## part a
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def Design_Matrix_X(x, y, n):
	N = len(x)
	l = int((n+1)*(n+2)/2)		
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k

	return X

n_x=1000
m=5

x = np.random.uniform(0, 1, n_x)
y = np.random.uniform(0, 1, n_x)

z = FrankeFunction(x, y)

#print(x)

n = int(len(x))
z_1 = z +0.01*np.random.randn(n)

X= Design_Matrix_X(x,y,n=m)
DesignMatrix = pd.DataFrame(X)
#print(DesignMatrix)

a = np.linalg.matrix_rank(X) #we check it is not a singular matrix
#print(a)

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_1)
ztilde = X @ beta
#print(beta)

beta1 = skl.LinearRegression().fit(X,z_1) #function .fit fits linear models
ztilde1 = beta1.predict(X)

#print(ztilde)
#print('--')
#print(ztilde1)

var_beta_OLS = 1*np.linalg.inv(X.T.dot(X))
var = pd.DataFrame(var_beta_OLS)
#print(var)
var_diag=np.diag(var_beta_OLS)
#print(var_diag)

l1_OLS = beta - 1.96*np.sqrt(var_diag)/(X.shape[0])
l2_OLS = beta + 1.96*np.sqrt(var_diag)/(X.shape[0])
#print(l1_OLS)
#print(l2_OLS)

def MSE (ydata, ymodel):
    n = np.size(ymodel)
    y = (ydata - ymodel).T@(ydata - ymodel)
    y = y/n
    return y

def R2 (ydata, ymodel):
   return 1-((ydata-ymodel).T@(ydata-ymodel))/((ydata-np.mean(ydata)).T@(ydata-np.mean(ydata)))


print(MSE(z_1,ztilde))
print(R2(z_1,ztilde))


print("Mean squared error: %.2f" % mean_squared_error(z_1, ztilde))
print('Variance score: %.2f' % r2_score(z_1, ztilde))

## part b

def train_test_splitdata(x_,y_,z_,i):

	x_learn=np.delete(x_,i)
	y_learn=np.delete(y_,i)
	z_learn=np.delete(z_,i)
	x_test=np.take(x_,i)
	y_test=np.take(y_,i)
	z_test=np.take(z_,i)

	return x_learn,y_learn,z_learn,x_test,y_test,z_test

def k_fold(k,x,y,z,m,model):
    n=len(x)
    j=np.arange(n)
    np.random.shuffle(j)
    n_k=int(n/k)
    MSE_K_t = 0
    R2_K_t = 0
    Variance_t=0
    Bias_t=0
    betas = np.zeros((k,int((m+1)*(m+2)/2)))
    z_pred = np.zeros((200,k))
    z_test1 = np.zeros((200,k))
    z_train1 = np.zeros((800,k))
    z_pred_train = np.zeros((800,k))
    for i in range(k):
        x_l,y_l,z_l,x_test,y_test,z_test=train_test_splitdata(x,y,z,j[i*n_k:(i+1)*n_k])
        z_test1[:,i]=z_test
        z_train1[:,i]=z_l
        X = Design_Matrix_X(x_l,y_l,m)
        X_test= Design_Matrix_X(x_test,y_test,m)
        #print(pd.DataFrame(X))
        #print(pd.DataFrame(X_test))
        beta1= model.fit(X,z_l)
        beta = beta1.coef_
        print(beta[0])
        betas[i] = beta
        ztilde1 = beta1.predict(X_test)
        ztilde_l = beta1.predict(X)
        #print(ztilde1)
        z_pred[:,i] = ztilde1
        z_pred_train[:,i] = ztilde_l
       # MSE_K_t+=MSE(z_test,ztilde1)
        R2_K_t+=R2(z_test,ztilde1)
       # Bias_t+=bias(z_test,ztilde1)
       # Variance_t+=variance(ztilde1)
# check if the values computed with our function and using the methods in lines 161-163 are the same
    #error_t = MSE_K_t/k
    #bias_t = Bias_t/k
    #variance_t = Variance_t/k
    R2_t = R2_K_t/k
    #print(error_t)
    #print(bias_t)
    #print(variance_t)

    error_test = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
    bias___ = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance___ = np.mean( (z_pred - np.mean(z_pred, axis=1, keepdims=True))**2 )
    error_train = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
    
    return (error_test, bias___,variance___ , error_train, R2_t, np.std(betas, axis = 0), np.mean(betas, axis = 0))


def variance(y_tilde):
	return np.sum((y_tilde - np.mean(y_tilde))**2)/np.size(y_tilde)

def bias(y, y_tilde):
	return np.sum((y - np.mean(y_tilde))**2)/np.size(y_tilde)

a=k_fold(5,x,y,z_1,5,LinearRegression(fit_intercept=False))
error_test = a[0]
bias___ = a[1]
variance___ = a[2]
error_train = a[3]
print('{} = {} + {}= {}'.format(error_test, bias___, variance___, bias___+variance___))


print('BBB')
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
kfold = model_selection.KFold(n_splits=5, shuffle=True)
X= Design_Matrix_X(x,y,n=5)
k=5
z_pred = []
z_test1 = []
z_train1 = []
z_pred_train = []
for train_index, test_index in kfold.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    z_train, z_test = z[train_index], z[test_index]
    z_test1.append(z_test)
    z_train1.append(z_train)
    print(X_train.shape, X_test.shape)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train,z_train)
    z_pred.append(model.predict(X_test))
    z_pred_train.append(model.predict(X_train))
bias = np.mean( (z_test - np.mean(z_pred))**2 )
variance = np.mean( (z_pred - np.mean(z_pred))**2 )
mse = model_selection.cross_val_score(model, X, z_1, cv=kfold, scoring='neg_mean_squared_error')
r2 = model_selection.cross_val_score(model, X, z_1, cv=kfold, scoring='r2')
print(bias)
print(variance)
print(np.absolute(mse.mean()))
print(r2.mean())


# part c

maxdegree = 20

def fold_degree(maxdegree,x,y,z,k):
    error__t = np.zeros(maxdegree)
    bias__t = np.zeros(maxdegree)
    variance__t = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)
    var_score__t = np.zeros(maxdegree)
    error__l = np.zeros(maxdegree)
    for degree in range(maxdegree):
        #z_pred = np.empty((2000, k))
        degree_fold = k_fold(k, x, y, z, degree, LinearRegression())
        error_t = degree_fold[0]
        bias_t = degree_fold[1]
        variance_t = degree_fold[2]
        var_score_t = degree_fold[4]
        error_l = degree_fold[3]
        polydegree[degree] = degree
        error__t[degree] = error_t
        bias__t[degree] = bias_t
        variance__t[degree] = variance_t
        var_score__t[degree] = var_score_t
        error__l[degree] = error_l
        print(degree)
        print(error_t)
        print(variance_t)
    return (polydegree, error__t, bias__t, variance__t, var_score__t, error__l)

b = fold_degree(maxdegree, x, y, z, 5)
#print(b[1])
#print(b[2], b[3])
#print(b[1]+b[3])

plt.plot(b[0], (b[1]), label='Error')
plt.plot(b[0], (b[2]), label='bias')
plt.plot(b[0], (b[3]), label='Variance')
plt.legend()
plt.show()

plt.plot(b[0], (b[1]), label='Error test')
plt.plot(b[0], (b[5]), label='Error learning')
plt.legend()
plt.show()

from sklearn.utils import resample

n_boostraps = 100

error_test = np.zeros(maxdegree)
bias___ = np.zeros(maxdegree)
variance___ = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
error_train = np.zeros(maxdegree)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
z_test1 = np.zeros((200,100))
z_train1 = np.zeros((800,100))
for i in range(100):
    z_test1[:,i]=z_test

for degree in range(maxdegree):
    model = LinearRegression(fit_intercept=False)
    z_pred = np.empty((z_test.shape[0],n_boostraps))
    z_pred_train = np.empty((z_train.shape[0],n_boostraps))
    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        z_train1[:,i] = z_
        X_train = Design_Matrix_X(x_,y_,degree)
        X_test= Design_Matrix_X(x_test,y_test,degree)  
        z_pred[:, i] = model.fit(X_train, z_).predict(X_test).ravel()
        z_pred_train[:, i] = model.fit(X_train, z_).predict(X_train).ravel()
    
    polydegree[degree] = degree
    error_test[degree] = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
    bias___[degree] = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance___[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True))
    error_train[degree] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
    #print(degree)
    #print(error_test)
    #print(bias___)
    #print(variance___)
    #print(bias___+variance___)
    

plt.plot(polydegree, error_test, label='Error')
plt.plot(polydegree, bias___, label='bias')
plt.plot(polydegree, variance___, label='Variance')
plt.legend()
plt.show()

plt.plot(polydegree, error_test, label='Error test')
plt.plot(polydegree, error_train, label='error training')
plt.legend()
plt.show()

#part d

lamdas = [0.001, 0.01, 0.1, 1]

for lamda in lamdas:
    beta_r = np.linalg.inv(X.T.dot(X)+lamda*np.identity(21)).dot(X.T).dot(z_1)
    zridge = X @ beta_r
    print("Beta parameters") 
    print(beta_r)
#print(zridge)

    clf_ridge = skl.Ridge(alpha=lamda).fit(X, z_1)
    zridge1 = clf_ridge.predict(X)
#print(zridge1)

    M = np.linalg.inv(X.T.dot(X)+lamda*np.identity(21))
    var_beta_ridge = M.dot(X.T).dot(X).dot(M.T)
    var_b_ridge = np.diag(var_beta_ridge)
    print("Variance of betas")
    print(var_b_ridge)

    l1_Ridge = beta_r - 1.96*np.sqrt(var_b_ridge)/(X.shape[0])
    l2_Ridge = beta_r + 1.96*np.sqrt(var_b_ridge)/(X.shape[0])
#print(l1_Ridge)
#print(l2_Ridge)

    print(MSE(z_1,zridge))
    print(R2(z_1,zridge))

    c = k_fold(5,x,y,z,5,skl.Ridge(alpha=lamda))
#print(c[0])
#print(c[1])
#print(c[2])
#print(c[3])
    


def fold_degree_r(x,y,z,k,lamdas):
    error = np.zeros(len(lamdas))
    bias = np.zeros(len(lamdas))
    variance = np.zeros(len(lamdas))
    polylamda = np.zeros(len(lamdas))
    for lamda in lamdas: 
        lamda_fold = k_fold(k, x, y, z, 5, skl.Ridge(alpha=lamda))
        error_ = lamda_fold[0]
        bias_ = lamda_fold[2]
        #print(bias_)
        variance_ = lamda_fold[3]
       # print('AAA')
        #print(lamdas.index(lamda))
        polylamda[lamdas.index(lamda)] = lamda
        error[lamdas.index(lamda)] = error_
        bias[lamdas.index(lamda)] = bias_
        variance[lamdas.index(lamda)] = variance_
    return (polylamda, error, bias, variance)

d = fold_degree_r(x, y, z, 5, lamdas)
#print(b[2])

plt.plot(d[0], d[1], label='Error')
plt.plot(d[0], d[2], label='bias')
plt.plot(d[0], d[3], label='Variance')
plt.legend()
plt.show()

n_boostraps = 100

error_test = np.zeros(len(lamdas))
bias___ = np.zeros(len(lamdas))
variance___ = np.zeros(len(lamdas))
polylamda = np.zeros(len(lamdas))
error_train = np.zeros(len(lamdas))
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
z_test1 = np.zeros((200,100))
z_train1 = np.zeros((800,100))
for i in range(100):
    z_test1[:,i]=z_test

for lamda in lamdas:
    model = skl.Ridge(alpha=lamda)
    z_pred = np.empty((z_test.shape[0],n_boostraps))
    z_pred_train = np.empty((z_train.shape[0],n_boostraps))
    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        z_train1[:,i] = z_
        X_train = Design_Matrix_X(x_,y_,5)
        X_test= Design_Matrix_X(x_test,y_test,5)  
        z_pred[:, i] = model.fit(X_train, z_).predict(X_test).ravel()
        z_pred_train[:, i] = model.fit(X_train, z_).predict(X_train).ravel()
    
    polylamda[lamdas.index(lamda)] = lamda
    error_test[lamdas.index(lamda)] = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
    bias___[lamdas.index(lamda)] = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance___[lamdas.index(lamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
    error_train[lamdas.index(lamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
    print(lamda)
    print(error_test)
    print(bias___)
    print(variance___)
    print(bias___+variance___)
    

plt.plot(lamdas, error_test, label='Error')
plt.plot(lamdas, bias___, label='bias')
plt.plot(lamdas, variance___, label='Variance')
plt.legend()
plt.show()

plt.plot(lamdas, error_test, label='Error test')
plt.plot(lamdas, error_train, label='error training')
plt.legend()
plt.show()


# part e)

lamda=0.01
model_lasso = skl.Lasso(alpha=lamda).fit(X, z_1)
betas = model_lasso.coef_
zlasso = model_lasso.predict(X)
print(MSE(z_1,zlasso))
print(R2(z_1,zlasso))
    
e = k_fold(5,x,y,z,5,skl.Lasso(alpha=lamda))    
print(e[0])

lamdas = [0.001, 0.01, 0.1, 1]

def fold_degree_r(x,y,z,k):
    lamdas = [0.001, 0.01, 0.1, 1]
    error = np.zeros(len(lamdas))
    bias = np.zeros(len(lamdas))
    variance = np.zeros(len(lamdas))
    polylamda = np.zeros(len(lamdas))
    for lamda in lamdas: 
        lamda_fold = k_fold(k, x, y, z, 5, skl.Lasso(alpha=lamda))
        error_ = lamda_fold[0]
        bias_ = lamda_fold[2]
        #print(bias_)
        variance_ = lamda_fold[3]
       # print('AAA')
        #print(lamdas.index(lamda))
        polylamda[lamdas.index(lamda)] = lamda
        error[lamdas.index(lamda)] = error_
        bias[lamdas.index(lamda)] = bias_
        variance[lamdas.index(lamda)] = variance_
    return (polylamda, error, bias, variance)

f = fold_degree_r(x, y, z, 5)
print(f[1], f[2])

plt.plot(f[0], f[1], label='Error')
plt.plot(f[0], f[2], label='bias')
plt.plot(f[0], f[3], label='Variance')
plt.legend()
plt.show()

n_boostraps = 100

error_test = np.zeros(len(lamdas))
bias___ = np.zeros(len(lamdas))
variance___ = np.zeros(len(lamdas))
polylamda = np.zeros(len(lamdas))
error_train = np.zeros(len(lamdas))
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, shuffle=True)
z_test1 = np.zeros((200,100))
z_train1 = np.zeros((800,100))
for i in range(100):
    z_test1[:,i]=z_test

for lamda in lamdas:
    model = skl.Lasso(alpha=lamda)
    z_pred = np.empty((z_test.shape[0],n_boostraps))
    z_pred_train = np.empty((z_train.shape[0],n_boostraps))
    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        z_train1[:,i] = z_
        X_train = Design_Matrix_X(x_,y_,5)
        X_test= Design_Matrix_X(x_test,y_test,5)  
        z_pred[:, i] = model.fit(X_train, z_).predict(X_test).ravel()
        z_pred_train[:, i] = model.fit(X_train, z_).predict(X_train).ravel()
    
    polylamda[lamdas.index(lamda)] = lamda
    error_test[lamdas.index(lamda)] = np.mean(np.mean((z_test1 - z_pred)**2 , axis=1, keepdims=True))
    bias___[lamdas.index(lamda)] = np.mean( (z_test1 - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance___[lamdas.index(lamda)] = np.mean( np.var(z_pred, axis=1, keepdims=True))
    error_train[lamdas.index(lamda)] = np.mean(np.mean((z_train1 - z_pred_train)**2 , axis=1, keepdims=True))
    print(lamda)
    print(error_test)
    print(bias___)
    print(variance___)
    print(bias___+variance___)
    

plt.plot(error_test, label='Error')
plt.semilogx(lamdas, error_test)
print(lamdas)
print(error_test)
plt.xlabel('lamdas')
plt.plot(lamdas, bias___, label='bias')
plt.plot(lamdas, variance___, label='Variance')
plt.legend()
plt.show()

plt.plot(lamdas, error_test, label='Error test')
plt.plot(lamdas, error_train, label='error training')
plt.legend()
plt.show()