from sklearn.datasets import load_iris
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegressionCV
X, y = datasets.make_moons(200, noise=0.20)
#X, y = load_iris(return_X_y=True)
clf = LogisticRegressionCV(cv=5, random_state=0,multi_class='multinomial').fit(X, y)
#clf.predict(X[:2, :])
#clf.predict_proba(X[:2, :]).shape
print(clf.score(X, y) )
