import numpy as np
from sklearn import linear_model, metrics 
from sklearn.model_selection import train_test_split

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
        self.regression = linear_model.Lasso(fit_intercept=True, max_iter=1000000, alpha=self.lambda_)
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
        self.ols = linear_model.LinearRegression().fit(self.X_train, self.y_train)
        self.beta = self.ols.coef_
        self.beta[0] = self.ols.intercept_
    
    def _SKLlassoFit(self, X_train, y_train, lambda_):
        self.regression = linear_model.Lasso(fit_intercept=True, max_iter=100000, alpha=self.lambda_)
        self.regression.fit(X_train,y_train)
        self.beta = self.regression.coef_
        self.beta[0] = self.regression.intercept_ 
    
    def _sklRidgeFit(self, X_train, y_train, lambda_):
        self.regression = linear_model.Ridge(fit_intercept=True, alpha=self.lambda_)
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
        self._MSE = metrics.mean_squared_error(self.y_test, self.yHat)
        return self._MSE
    
    def R2(self, y_test):
        self.y_test = y_test
        if self.yHat is None :
            self._sklPredict()
        self._R2 = metrics.r2_score(self.y_test, self.yHat)
        return self._R2