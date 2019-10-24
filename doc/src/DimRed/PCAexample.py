from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X_train_scaled)

X_pca = pca.transform(X_train_scaled)

X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]

W2 = V.T[:, :2]
X2D = X_centered.dot(W2)

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

pca.components_.T[:, 0]).

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)


import numpy as np
import pandas as pd
from IPython.display import display
np.random.seed(100)
# setting up a 10 x 5 matrix
rows = 10
cols = 5
a = np.random.randn(rows,cols)
df = pd.DataFrame(a)
display(df)
print(df.mean())
print(df.std())
display(df**2)


