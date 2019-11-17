import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer

# Load the data
cancer = load_breast_cancer()
maxdepth = 6
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
score = np.zeros(maxdepth)
depth = np.zeros(maxdepth)

for degree in range(1,maxdepth):
    model = DecisionTreeRegressor(max_depth=degree) 
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    depth[degree] = degree
    score[degree] = model.score(X_test_scaled,y_pred)
    print('Max Tree Depth:', degree)
    print('Score:', score[degree])

plt.xlim(1,maxdepth)
plt.plot(depth, score, label='Score')
plt.legend()
plt.show()



