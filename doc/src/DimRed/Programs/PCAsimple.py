import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

n = 1000
mean = (-1, 2)
cov = [[10, 1], [1, 0.5]]
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
x = X_centered[:,0]
y = X_centered[:,1]
Cov = np.zeros((2,2))
Cov[0,1] = np.sum(x.T@y)/(n-1.0)
Cov[0,0] = np.sum(x.T@x)/(n-1.0)
Cov[1,1] = np.sum(y.T@y)/(n-1.0)
Cov[1,0]= Cov[0,1]
print("Centered covariance using own code")
print(Cov)

plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()

# diagonalize and obtain eigenvalues, not necessarily sorted
EigValues, EigVectors = np.linalg.eig(Cov)
# sort eigenvectors and eigenvalues
#permute = EigValues.argsort()
#EigValues = EigValues[permute]
#EigVectors = EigVectors[:,permute]
print("Eigenvalues of Covariance matrix")
for i in range(2):
    print(EigValues[i])
FirstEigvector = EigVectors[:,0]
SecondEigvector = EigVectors[:,1]
print("First eigenvector")
print(FirstEigvector)
print("Second eigenvector")
print(SecondEigvector)
#thereafter we do a PCA with Scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2Dsl = pca.fit_transform(X)
print("Eigenvector of largest eigenvalue")
print(pca.components_.T[:, 0])



