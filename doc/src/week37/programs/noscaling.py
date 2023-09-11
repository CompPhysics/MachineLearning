import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def OLS_fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def Ridge_fit_beta(X, y,L,d):
    I = np.eye(d,d)
    return np.linalg.pinv(X.T @ X + L*I) @ X.T @ y


np.random.seed(2018)
n = 100
d = 3
L = 0.001
true_beta = [2, 0.5, 3.7]

# Make data set.
x = np.linspace(-3, 3, n)
y_real = 2 + 0.5*x + 3.7*x**2

y = np.sum(
    np.asarray([x ** p * b for p, b in enumerate(true_beta)]), 
    axis=0) + 0.1 * np.random.normal(size=len(x))


#Design matrix X including the intercept
X = np.zeros((len(x), d))
for p in range(d):     # (d-1)
    X[:, p] = x ** (p) # (p+1 if not intercept included)


#Split datamatrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Calculate beta, own code
beta_OLS = OLS_fit_beta(X_train, y_train)
beta_Ridge = Ridge_fit_beta(X_train, y_train,L,d)
print(beta_OLS)
print(beta_Ridge)

#predict value
ytilde_test_OLS = X_test @ beta_OLS
ytilde_test_Ridge = X_test @ beta_Ridge


#Calculate MSE

print("  ")
print("test MSE of OLS:")
print(MSE(y_test,ytilde_test_OLS))
print("  ")
print("test MSE of Ridge")
print(MSE(y_test,ytilde_test_Ridge))


plt.scatter(x,y,label='Data')
#plt.plot(x,y_real,label='no noise')
plt.plot(x, X @ beta_OLS,'*', label="OLS_Fit")
plt.plot(x, X @ beta_Ridge, label="Ridge_Fit")
plt.grid()
plt.legend()
plt.show()
