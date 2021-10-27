Aimport os
import sys
import pytest
import numba
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import functools
import time
from numba import jit, njit
from PIL import Image
import pandas as pd
import seaborn as sns
sns.set()
import math

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn import datasets
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.utils import resample


# Bootstrap

def Bootstrap(x1,x2, y, N_boot=500, method = 'ols', degrees = 5, random_state = 42):
    """
    Computes bias^2, variance and the mean squared error using bootstrap resampling method
    for the provided data and the method.
    
    Arguments:
    x1: 1D numpy array, covariate
    x2: 1D numpy array, covariate
    N_boot: integer type, the number of bootstrap samples
    method: string type, accepts 'ols', 'ridge' or 'lasso' as arguments
    degree: integer type, polynomial degree for generating the design matrix
    random_state: integer, ensures the same split when using the train_test_split functionality
    
    Returns: Bias_vec, Var_vec, MSE_vec, betaVariance_vec
             numpy arrays. Bias, Variance, MSE and the variance of beta for the predicted model
    """
    ##split x1, x2 and y arrays as a train and test data and generate design matrix
    x1_train, x1_test,x2_train, x2_test, y_train, y_test = train_test_split(x1,x2, y, test_size=0.2, random_state = random_state)
    y_pred_test = np.zeros((y_test.shape[0], N_boot))
    X_test = designMatrix(x1_test, x2_test, degrees)
    
    betaMatrix = np.zeros((X_test.shape[1], N_boot))
    
    ##resample and fit the corresponding method on the train data
    for i in range(N_boot):
        x1_,x2_, y_ = resample(x1_train, x2_train, y_train)
        X_train = designMatrix(x1_, x2_, degrees)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_train[:, 0] = 1
        X_test = designMatrix(x1_test, x2_test, degrees)
        X_test = scaler.transform(X_test)
        X_test[:, 0] = 1
        
        if method == 'ols':
            manual_regression = linregOwn(method = 'ols')
            beta =  manual_regression.fit(X_train, y_)
        if method == 'ridge':
            manual_regression = linregOwn(method = 'ridge')
            beta =  manual_regression.fit(X_train, y_, lambda_ = 0.05)
        if method == 'lasso':
            manual_regression = linregOwn(method = 'lasso')
            beta =  manual_regression.fit(X_train, y_, lambda_ = 0.05)
            
        ##predict on the same test data
        y_pred_test[:, i] = np.dot(X_test, beta)
        betaMatrix[:, i] = beta
    y_test = y_test.reshape(len(y_test),1) 
      
    Bias_vec = []
    Var_vec  = []
    MSE_vec  = []
    betaVariance_vec = []
    R2_score = []
    y_test = y_test.reshape(len(y_test),1)
    MSE = np.mean( np.mean((y_test - y_pred_test)**2, axis=1, keepdims=True) )
    bias = np.mean( (y_test - np.mean(y_pred_test, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred_test, axis=1, keepdims=True) )
    betaVariance = np.var(betaMatrix, axis=1)
    print("-------------------------------------------------------------")
    print("Degree: %d" % degrees)
    print('MSE:', np.round(MSE, 3))
    print('Bias^2:', np.round(bias, 3))
    print('Var:', np.round(variance,3))
    print('{} >= {} + {} = {}'.format(MSE, bias, variance, bias+variance))
    print("-------------------------------------------------------------")
    
    Bias_vec.append(bias)
    Var_vec.append(variance)
    MSE_vec.append(MSE)
    betaVariance_vec.append(betaVariance)
    return Bias_vec, Var_vec, MSE_vec, betaVariance_vec



class CrossValidation:
    """  
        A class of cross-validation technique. Performs cross-validation with shuffling.
    """
    def __init__(self, LinearRegression, DesignMatrix):
        """
        Initialization
                
        Arguments:
        LinearRegression: Instance from the class created by either linregOwn or linregSKl
        DesignMatrix: Function that generates design matrix 
        """
        self.LinearRegression = LinearRegression
        self.DesignMatrix = DesignMatrix
    
    def kFoldCV(self, x1, x2, y, k = 10, lambda_ = 0, degree = 5):
        """
        Performs shuffling of the data, holds a split of the data as a test set at each split and evaluates the model
        on the rest of the data. 
        Calculates the MSE , R2_score, variance, bias on the test data and MSE on the train data.
        
        Arguments:
        x1: 1D numpy array
        x2: 1D numpy array
        y: 1D numpy array
        k: integer, the number of splits
        lambda_: float type, shrinkage parameter for ridge and lasso methods.
        degree: integer type, the number of polynomials, complexity parameter
        
        """
        self.lambda_ = lambda_
        M = x1.shape[0]//k   ## Split input data x in k folds of size M

        
        ##save the statistic in the list
        MSE_train = []
        MSE_k     = []
        R2_k      = []
        var_k     = []
        bias_k    = []
        
        ##shuffle the data randomly
        shf = np.random.permutation(x1.size)
        x1_shuff = x1[shf]
        x2_shuff = x2[shf]
        y_shuff = y[shf]
        
        for i in range(k):
            # x_k and y_k are the hold out data for fold k
            x1_k = x1_shuff[i*M:(i+1)*M]
            x2_k = x2_shuff[i*M:(i+1)*M]
            y_k = y_shuff[i*M:(i+1)*M]
            
            ## Generate train data and then scale both train and test
            index_true = np.array([True for i in range(x1.shape[0])])
            index_true[i*M:(i+1)*M] = False
            X_train = self.DesignMatrix(x1_shuff[index_true], x2_shuff[index_true], degree)
            y_train = y_shuff[index_true]
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_train[:, 0] = 1
 
            ### Fit the regression on the train data
            beta = self.LinearRegression.fit(X_train, y_train, lambda_)
            y_predict_train = np.dot(X_train, beta)
            MSE_train.append(np.sum( (y_train-y_predict_train)**2)/len(y_train))
            
            ## Predict on the hold out data and calculate statistic of interest
            X_k = self.DesignMatrix(x1_k, x2_k, degree)
            X_k = scaler.transform(X_k)
            X_k[:, 0] = 1
            y_predict = np.dot(X_k,beta)
            MSE_k.append(np.sum((y_k-y_predict)**2, axis=0, keepdims=True)/len(y_predict))
            R2_k.append(1.0 - np.sum((y_k - y_predict)**2, axis=0, keepdims=True) / np.sum((y_k - np.mean(y_k))**2, axis=0, keepdims=True) )
            var_k.append(np.var(y_predict,axis=0, keepdims=True))
            bias_k.append((y_k - np.mean(y_predict, axis=0, keepdims=True))**2 )        
        
        means = [np.mean(MSE_k), np.mean(R2_k), np.mean(var_k), 
                 np.mean(bias_k),np.mean(MSE_train)]
        #print('MSE_test: {}' .format(np.round(np.mean(MSE_k),3)))
        #print('R2: {}' .format(np.round(np.mean(R2_k),3)))
        #print('Variance of the predicted outcome: {}' .format(np.round(np.mean(var_k),3)))
        #print('Bias: {}' .format(np.round(np.mean(bias_k),3)))
        #print('MSE_train {}' .format(np.round(np.mean(MSE_train),3)))
        return means


# Franke Function

def franke(x, y):
    """ 
    Computes Franke function. 
    Franke's function has two Gaussian peaks of different heights, and a smaller dip. 
    It is used as a test function in interpolation problems.
    
    Franke's function is normally defined on the grid [0, 1] for each x, y.
    
    Arguments of the function:
    x : numpy array
    y : numpy array
    
    Output of the function:
    f : Franke function values at specific coordinate points of x and y
    """
    f = (0.75 * np.exp(-((9*x - 2)**2)/4  - ((9*y - 2)**2)/4 ) 
        + 0.75 * np.exp(-((9*x + 1)**2)/49 -  (9*y + 1)    /10) 
        + 0.5  * np.exp(-((9*x - 7)**2)/4  - ((9*y - 3)**2)/4 ) 
        - 0.2  * np.exp(-((9*x - 4)**2)    - ((9*y - 7)**2)   ))
    return f




class linregOwn:
    """
    A class of linear regressions. Perform ordinarly least squares (OLS) and Ridge regression manually. Lasso
    is performed using scikit-learn functionality.
    """
    def __init__(self, method = 'ols'):
        """
        Constructor
        
        Determines the method used in the fitting
        
        Arguments:
        method: string type. Accepts either 'ols', 'ridge' or 'lasso'.
        
        """
        self.method = method
        self.yHat           = None
        self.X              = None
        self.y              = None
        self.beta           = None
        
        self._MSE           = None
        self._R2            = None
        self._betaVariance  = None
        self.lambda_        = None
        
    def fit(self, X_train, y_train, lambda_ = 0):
        """
        Performs the fit of OLS, Ridge or Lasso, depending on the argument provided initially.
        
        Arguments:
        X_train: Covariate matrix of the train data set, i.e. design matrix of 
                the shape m x p where m is the number of rows and p is the number of columns  
                (i.e. p is the complexity parameter).
        y_train: Outcome variable, 1D numpy array
        lambda_: float type. Shrinkage parameter for ridge and lasso methods. The higher value, higher shrinkage.
                 lambda_ is set to 0 for the OLS regression 
                 
        """
        self.X_train = X_train
        self.y_train = y_train
        self.lambda_ = lambda_
        if self.method == 'ols':
            self._olsFit(X_train, y_train)
        if self.method == 'ridge':
            self._ridgeFit(X_train, y_train, lambda_)
        if self.method == 'lasso':
            self._lassoFitSKL(X_train, y_train, lambda_)
        return self.beta
        
    def _olsFit(self, X_train, y_train):
        """
        Performs the ordinary least squares (OLS) fit on the provided data using singular value decomposition(SVD).
        
        
        Arguments:
        
        X_train: Covariate matrix of the train data set, i.e. design matrix of 
                the shape m x p where m is the number of rows and p is the number of columns  
                (i.e. p is the complexity parameter).
        y_train: Outcome variable, 1D numpy array
        
        Returns:
            beta : numpy.array
            The beta parameters from the performed fit
        """
        self.X_train = X_train
        self.y_test = y_train
        U, S, VT = np.linalg.svd(self.X_train, full_matrices=True)
        S_inverse = np.zeros(shape=self.X_train.shape)
        ##S is a vector, with shape of the number of columns
        S_inverse[:S.shape[0], :S.shape[0]] = np.diag(1/S)
        self.beta = np.dot(VT.T, np.dot(S_inverse.T, np.dot(U.T, self.y_train)))
        #self.beta = np.linalg.inv(np.dot(X.T,X)).dot(X.T, y)
        
    def _ridgeFit(self, X_train, y_train, lambda_):
        """
        Performs the ridge regression fit
        
        Arguments:
        X_train: Covariate matrix of the train data set, design matrix of 
                the shape m x p (m_train_rows, p_columns).
        y_train: Outcome variable, 1D numpy array, dimension m x 1 
        lambda_: Integer type. The shrinkage parameter 
        
        Returns:
            beta : numpy.array
            The beta parameters from the performed fit
        """
        self.X_train = X_train
        self.y_train = y_train
        self.lambda_ = lambda_
        self.beta = np.dot(np.linalg.inv(np.dot(X_train.T,X_train) + self.lambda_ * np.eye(X_train.shape[1])), np.dot(X_train.T,y_train))
    
    def _lassoFitSKL(self, X_train, y_train, lambda_):
        """
        Performs lasso fit using scikit-learn functionality. 
        
        Arguments:
        X_train: Covariate matrix of the train data set, design matrix of 
                the shape m x p (m_train_datapoints, p_parameters).
        y_train: Outcome variable, 1D numpy array, dimension m x 1 
        lambda_: Integer type. The shrinkage parameter 
        
        Returns:
            self.beta : numpy.array
            The beta parameters from the performed fit
        """
        self.regression = Lasso(fit_intercept=True, max_iter=1000000, alpha=self.lambda_)
        self.regression.fit(X_train,y_train)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_ 
        
    def predict(self, X_test):
        """
        Performs prediction of the fitted model on the provided test data set.
        
        Arguments:
        X_test: Design matrix, covariate matrix, dimension k  x  p (k_test_rows, p_columns)
        
        Returns: self.yHat
                 numpy 1D array, prediction values of dimension k x p 
        """
        self.X_test = X_test
        self._predictOwntest(X_test)
        return self.yHat
        
    def _predictOwntest(self, X_test):
        """
        Performs manual prediction of the given model on the train data.
        """
        self.X_test = X_test
        self.yHat = np.dot(self.X_test, self.beta)
        
    def MSE(self, y_test):
        """
        Calculates the mean squared error (MSE) manually after the fit and prediction have been implemented.
        
        Arguments:
        y_test: Outcome variable, 1D numpy array, dimension k x 1 (k_test_rows, 1_column)
        
        Returns: self._MSE
                 The mean squared error of the predicted model
        """
        self.y_test = y_test
        if self.yHat is None :
            self._predictOwntest(X_test)
        N = self.yHat.size
        self._MSE = (np.sum((self.y_test - self.yHat)**2))/N
        return self._MSE
    
    def R2(self, y_test):
        """
        Calculates R2 score manually after the fit and prediction have been implemented.
        
        Arguments:
        y_test: Outcome variable, 1D numpy array, dimension k x 1 (k_test_rows, 1_column)
        
        Returns: self._R2
                 The R2 score of the predicted model
        """
        self.y_test = y_test
        if self.yHat is None:
            self._predictOwntest(X_test)
        yMean = (1.0 / self.y_test.size) * np.sum(self.y_test)
        self._R2 = 1.0 - np.sum((self.y_test - self.yHat)**2) / np.sum((self.y_test - yMean)**2)
        return  self._R2
    
    def CI(self, y_test):
        """
        Calculates confidence intervals manually after the fit and prediction have been implemented.
        
        Arguments:
        y_test: Outcome variable, 1D numpy array, dimension k x 1 (k_test_rows, 1_column)
        
        Returns: var, Lower, Upper
                 Variance, Lower and Upper bounds of the confidence intervals for the parameter self.beta
        """  
        self.y_test = y_test
        if self.yHat is None:
            self._predictOwntest(X_test)
        sigma2 = np.sum(((self.y_test - self.yHat)**2))/(self.y_test.size - self.beta.size)
        var = np.diag(np.linalg.inv(np.dot(self.X_test.T, self.X_test))) * sigma2
        Lower = self.beta - 1.96*np.sqrt(var)
        Upper = self.beta + 1.96*np.sqrt(var)
        return var, Lower, Upper
 
 
 ###Implementation through scikitlearn      
class linregSKL:
    def __init__(self, method = 'ols'):
        """
        A class of linear regressions. Perform ordinarly least squares (OLS) and Ridge and Lasso
        using scikit-learn functionality. 
        
        """
        self.method = method
        self.yHat           = None
        self.X              = None
        self.y              = None
        self.beta           = None
        
        self._MSE           = None
        self._R2            = None
        self._betaVariance  = None

        
    def fit(self, X_train, y_train, lambda_ = 0):
        self.X_train = X_train
        self.y_train = y_train
        if self.method == 'ols':
            self._olsSKLfit(X_train, y_train)
        if self.method == 'ridge':
            self._sklRidgeFit(X_train, y_train, lambda_)
        if self.method == 'lasso':
            self._SKLlassoFit(X_train, y_train, lambda_)
        return self.beta
    
    def _olsSKLfit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        ##We already have standardized data from design matrix
        self.ols = LinearRegression().fit(self.X_train, self.y_train)
        self.beta = self.ols.coef_
        self.beta[0] = self.ols.intercept_
    
    def _SKLlassoFit(self, X_train, y_train, lambda_):
        self.regression = Lasso(fit_intercept=True, max_iter=100000, alpha=self.lambda_)
        self.regression.fit(X_train,y_train)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_ 
    
    def _sklRidgeFit(self, X_train, y_train, lambda_):
        self.regression = Ridge(fit_intercept=True, alpha=self.lambda_)
        self.regression.fit(X,y)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_
     
    def predict(self, X_test):
        self.X_test = X_test
        if self.method == 'ols':
            self._sklPredict(X_test)
        return self.yHat
               
    def _sklPredict(self, X_test):
        self.X_test = X_test
        ## Since our data contains 1-s, we should subtract intercept, since scikit learn additionally
        ##generates the 1-s
        self.yHat = self.ols.predict(self.X_test) - self.beta[0]
        
    def MSE(self, y_test):
        self.y_test = y_test
        if self.yHat is None :
            self._sklPredict(X_test)
        self._MSE = mean_squared_error(self.y_test, self.yHat)
        return self._MSE
    
    def R2(self, y_test):
        self.y_test = y_test
        if self.yHat is None :
            self._sklPredict()
        self._R2 = r2_score(self.y_test, self.yHat)
        return self._R2



def designMatrix(x, y, k=5):
    """
    Generates the design matrix (covariates of polynomial degree k). 
    Intercept is included in the design matrix. 
    Scaling does not apply to the intercept term.
    if k = 2, generated column vectors: 1, x, y, x^2, xy, y^2 
    if k = 3, generated column vectors: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    ...
    
    Arguments:
    x: 1D numpy array
    y: 1D numpy array
    k: integer type. complexity parameter (i.e polynomial degree) 
    """
    
    xb = np.ones((x.size, 1))
    
    for i in range(1, k+1):
        for j in range(i+1):
            xb = np.c_[xb, (x**(i-j))*(y**j)]

    xb[:, 0] = 1
    return xb




# Stochastic Gradient Descent

from matplotlib.ticker import LinearLocator, FormatStrFormatter

def compute_square_loss(X, y, theta):
    loss = 0 #Initialize the average square loss
    
    m = len(y)
    loss = (1.0/m)*(np.linalg.norm((X.dot(theta) - y)) ** 2)
    return loss


def gradient_ridge(X, y, beta, lambda_):
    return 2*(np.dot(X.T, (X.dot(beta) - y))) + 2*lambda_*beta

def gradient_ols(X, y, beta):
    m = X.shape[0]
    
    grad = 2/m * X.T.dot(X.dot(beta) - y)
    
    return grad

def learning_schedule(t):
    t0, t1 = 5, 50
    return t0/(t+t1)


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.random.permutation(inputs.shape[0])
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]


###sgd
def SGD(X, y, learning_rate = 0.02, n_epochs = 100, lambda_ = 0.01, batch_size = 20, method = 'ols'):
    num_instances, num_features = X.shape[0], X.shape[1]
    beta = np.random.randn(num_features) ##initialize beta
    
    for epoch in range(n_epochs+1):
        
        for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
             
            X_batch, y_batch = batch
            
            # for i in range(batch_size):
            #     learning_rate = learning_schedule(n_epochs*epoch + i)
            
            if method == 'ols':
                gradient = gradient_ols(X_batch, y_batch, beta)
                beta = beta - learning_rate*gradient
            if method == 'ridge':
                gradient = gradient_ridge(X_batch, y_batch, beta, lambda_ = lambda_)
                beta = beta - learning_rate*gradient
                
    mse_ols_train = compute_square_loss(X, y, beta) 
    mse_ridge_train = compute_square_loss(X, y, beta) + lambda_*np.dot(beta.T, beta)
            
    return beta

def compute_test_mse(X_test, y_test, beta, lambda_ = 0.01):
    mse_ols_test = compute_square_loss(X_test, y_test, beta) 
    mse_ridge_test = compute_square_loss(X_test, y_test, beta) + lambda_*np.dot(beta.T, beta)
    return mse_ols_test, mse_ridge_test   


# # Part A

# In[10]:


# a

##Make synthetic data
n = 1000  
np.random.seed(20)
x1 = np.random.rand(n)
x2 = np.random.rand(n)     
X = designMatrix(x1, x2, 4)
y = franke(x1, x2) 

##Train-validation-test samples. 
# We choose / play with hyper-parameters on the validation data and then test predictions on the test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

X_train[:, 0] = 1
X_test[:, 0] = 1
X_val[:, 0] = 1



linreg = linregOwn(method='ols')
#print('Invert OLS:', linreg.fit(X_train, y_train))
beta = SGD(X_train, y_train, learning_rate=0.07)
#print('SGD OLS:', beta)


linreg = linregOwn(method='ridge')
#print('Invert Ridge:', linreg.fit(X_train, y_train, lambda_= 0.01))
beta = SGD(X_train, y_train, learning_rate=0.0004, method='ridge')
#print('SGD Ridge:', beta)


sgdreg = SGDRegressor(max_iter = 100, penalty=None, eta0=0.1)
sgdreg.fit(X_train[:, 1:],y_train.ravel())
#print('sklearn:', sgdreg.coef_)
#print('sklearn intercept:', sgdreg.intercept_)


def plot_MSE(method = 'ridge', scheme = None):
    eta = np.logspace(-5, -3, 10)
    lambda_ = np.logspace(-5, -1, 10)
    MSE_ols = []
    MSE_ridge = []
    
    if scheme == 'joint':
        
        if method == 'ridge':
            
            for lmbd in lambda_:
                
                for i in eta:  
                    
                    beta = SGD(X_train, y_train, learning_rate=i, lambda_ = lmbd, method = method)
                    mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, lambda_ = lmbd, beta = beta)
                    MSE_ridge.append(mse_ridge_test)
            
            fig = plt.figure()
            ax = fig.gca(projection='3d') ##get current axis
            lambda_ = np.ravel(lambda_)
            eta = np.ravel(eta)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.03f'))
            ax.plot_trisurf(lambda_, eta, MSE_ridge, cmap='viridis', edgecolor='none')
            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel(r'$\eta$')
            ax.set_title(r'MSE Ridge')
            ax.view_init(30, 60)
            plt.show()
      
    if scheme == 'separate':
        
        if method == 'ols':
            
            eta = np.logspace(-5, 0, 10)  
            
            for i in eta:  
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = method)
                mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, beta = beta)
                MSE_ols.append(mse_ols_test)
            
            print('The learning rate {} performs best for the OLS' .format(eta[MSE_ols.index(min(MSE_ols))]))
            print('Corresponding minimum MSE for OLS: {}'.format(min(MSE_ols)))
            plt.semilogx(eta, MSE_ols)
            plt.xlabel(r'Learning rate, $\eta$')
            plt.ylabel('MSE OLS')
            plt.title('Stochastic Gradient Descent')
            plt.show()
    if scheme == 'separate':
        
        if method == 'ridge':
            
            eta = np.logspace(-5, 0, 10)  
            
            for i in eta:  
                
                beta = SGD(X_train, y_train, learning_rate=i, lambda_ = 0.01, method = method)
                mse_ols_test, mse_ridge_test = compute_test_mse(X_val, y_val, beta = beta)
                MSE_ols.append(mse_ridge_test)
            
            print('The learning rate {} performs best for Ridge' .format(eta[MSE_ols.index(min(MSE_ols))]))
            print('Corresponding minimum MSE for Ridge: {}'.format(min(MSE_ols)))

            plt.plot(eta, MSE_ols)
            plt.xlabel(r'Learning rate, $\eta$')
            plt.ylabel('MSE Ridge')
            plt.title('Stochastic Gradient Descent')
            plt.show()
        

 

####Predict OLS, Ridge on test data after tuning learning rate and lambda on validation data

def plot_scatter(y_true, method = 'ols'):
    if method == 'ols':
        beta = SGD(X_train, y_train, learning_rate=0.07, lambda_ = 0, method = method, n_epochs=300)
    if method == 'ridge':
        beta = SGD(X_train, y_train, learning_rate=0.0001, lambda_ = 0, method = method, n_epochs=300)
    y_pred = np.dot(X_test, beta)
    mse_ols_test, mse_ridge_test = compute_test_mse(X_test, y_true, beta = beta)
    print('Test MSE OLS: {}' .format(mse_ols_test))
    print('Test MSE Ridge: {}' .format(mse_ridge_test))
    a = plt.axes(aspect='equal')
    plt.scatter(y_pred, y_pred, color= 'blue', label = "True values")
    plt.scatter(y_pred, y_true, color = 'red', label = "Predicted values")
    plt.xlabel('True y values')
    plt.ylabel('Predicted y')
    plt.title(f"Prediction - {method}")
    plt.legend()
    # if method == 'ols':
    #     plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'ols_reg_pred.png'), transparent=True, bbox_inches='tight')
    # if method == 'ridge':
    #     plt.savefig(os.path.join(os.path.dirname(__file__), 'Plots', 'ridge_reg_pred.png'), transparent=True, bbox_inches='tight')
    
    plt.show()

plot_scatter(y_test, method='ols')

plot_scatter(y_test, method='ridge')



