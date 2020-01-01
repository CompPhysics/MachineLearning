import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler


n = 10000
mean = (-1, 2)
cov = [[4, 2], [2, 2]]
print(cov)
X = np.random.multivariate_normal(mean, cov, n)
print(np.cov(X.T))

df = pd.DataFrame(X)
# Pandas does the centering for us
df = df -df.mean()
correlation_matrix = df.cov()
print(correlation_matrix)

# we center it ourselves
X_centered = X - X.mean(axis=0)
print(np.cov(X_centered.T))
#print("test that we get the same as Pandas")
#print(X_centered-X_train_scaled)
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



