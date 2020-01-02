import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

n = 1000
mean = (-1, 2)
cov = [[4, 2], [2, 2]]
X = np.random.multivariate_normal(mean, cov, n)

df = pd.DataFrame(X)
# Pandas does the centering for us
df = df -df.mean()
print("Centered covariance with Pandas")
covarianceX = df.cov()

print(covarianceX)

# we center it ourselves
X_centered = X - X.mean(axis=0)
print("Centered covariance using numpy")
print(np.cov(X_centered.T))
# extract the relevant columns from the centered design matrix
x = X_centered[:,[0]]
y = X_centered[:,[1]]
Cov = np.zeros((2,2))
cov_xy = np.sum(x.T@y)/(n-1.0)
cov_xx = np.sum(x.T@x)/(n-1.0)
cov_yy = np.sum(y.T@y)/(n-1.0)

Cov[0,0]= cov_xx
Cov[1,1]= cov_yy
Cov[0,1]= cov_xy
Cov[1,0]= Cov[0,1]
print("Centered covariance using own code")
print(Cov)

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()




"""
#Now we do an SVD
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]
W2 = V.T[:, :2]
X2D = X_centered.dot(W2)
#thereafter we do a PCA with Scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2Dsl = pca.fit_transform(X)
print("Check that we get the same")
print(X2D-X2Dsl)

print(pca.components_.T[:, 0])
"""


