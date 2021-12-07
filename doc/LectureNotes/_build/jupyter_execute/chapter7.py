#!/usr/bin/env python
# coding: utf-8

# # Ensemble Methods: From a Single Tree to Many Trees and Extreme Boosting, Meet the Jungle of Methods
# 
# As stated previously and seen in many of the examples discussed in the previous chapter about
# a single decision tree, we often end up overfitting our training
# data. This normally means that we have a high variance. Can we reduce
# the variance of a statistical learning method?
# 
# This leads us to a set of different methods that can combine different
# machine learning algorithms or just use one of them to construct
# forests and jungles of trees, homogeneous ones or heterogenous
# ones. These methods are recognized by different names which we will
# try to explain here. These are
# 
# 1. Voting classifiers
# 
# 2. Bagging and Pasting
# 
# 3. Random forests
# 
# 4. Boosting methods, from adaptive to Extreme Gradient Boosting (XGBoost)
# 
# We discuss these methods here.
# 
# ### An Overview of Ensemble Methods
# 
# <!-- FIGURE: [DataFiles/ensembleoverview.png, width=600 frac=0.8] -->
# 
# 
# 
# ## Bagging
# 
# The **plain** decision trees suffer from high
# variance. This means that if we split the training data into two parts
# at random, and fit a decision tree to both halves, the results that we
# get could be quite different. In contrast, a procedure with low
# variance will yield similar results if applied repeatedly to distinct
# data sets; linear regression tends to have low variance, if the ratio
# of $n$ to $p$ is moderately large. 
# 
# **Bootstrap aggregation**, or just **bagging**, is a
# general-purpose procedure for reducing the variance of a statistical
# learning method. 
# 
# 
# Bagging typically results in improved accuracy
# over prediction using a single tree. Unfortunately, however, it can be
# difficult to interpret the resulting model. Recall that one of the
# advantages of decision trees is the attractive and easily interpreted
# diagram that results.
# 
# However, when we bag a large number of trees, it is no longer
# possible to represent the resulting statistical learning procedure
# using a single tree, and it is no longer clear which variables are
# most important to the procedure. Thus, bagging improves prediction
# accuracy at the expense of interpretability.  Although the collection
# of bagged trees is much more difficult to interpret than a single
# tree, one can obtain an overall summary of the importance of each
# predictor using the MSE (for bagging regression trees) or the Gini
# index (for bagging classification trees). In the case of bagging
# regression trees, we can record the total amount that the MSE is
# decreased due to splits over a given predictor, averaged over all $B$ possible
# trees. A large value indicates an important predictor. Similarly, in
# the context of bagging classification trees, we can add up the total
# amount that the Gini index  is decreased by splits over a given
# predictor, averaged over all $B$ trees.

# In[1]:


heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
save_fig("votingsimple")
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

log_clf = LogisticRegression(solver="liblinear", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
svm_clf = SVC(gamma="auto", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:


log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# ## Bagging Examples

# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[ ]:


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
save_fig("baggingtree")
plt.show()


# ### Making your own Bootstrap: Changing the Level of the Decision Tree
# 
# Let us bring up our good old boostrap example from the linear regression lectures. We change the linerar regression algorithm with
# a decision tree wth different depths and perform a bootstrap aggregate (in this case we perform as many bootstraps as data points $n$).

# In[ ]:



import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.tree import DecisionTreeRegressor

n = 100
n_boostraps = 100
maxdepth = 8

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
error = np.zeros(maxdepth)
bias = np.zeros(maxdepth)
variance = np.zeros(maxdepth)
polydegree = np.zeros(maxdepth)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# we produce a simple tree first as benchmark
simpletree = DecisionTreeRegressor(max_depth=3) 
simpletree.fit(X_train_scaled, y_train)
simpleprediction = simpletree.predict(X_test_scaled)
for degree in range(1,maxdepth):
    model = DecisionTreeRegressor(max_depth=degree) 
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(X_train_scaled, y_train)
        model.fit(x_, y_)
        y_pred[:, i] = model.predict(X_test_scaled)#.ravel()

    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))
 
mse_simpletree= np.mean( np.mean((y_test - simpleprediction)**2)
print(mse_simpletree)
plt.xlim(1,maxdepth)
plt.plot(polydegree, error, label='MSE')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
save_fig("baggingboot")
plt.show()


# ## Random forests
# 
# Random forests provide an improvement over bagged trees by way of a
# small tweak that decorrelates the trees. 
# 
# As in bagging, we build a
# number of decision trees on bootstrapped training samples. But when
# building these decision trees, each time a split in a tree is
# considered, a random sample of $m$ predictors is chosen as split
# candidates from the full set of $p$ predictors. The split is allowed to
# use only one of those $m$ predictors. 
# 
# A fresh sample of $m$ predictors is
# taken at each split, and typically we choose

# $$
# m\approx \sqrt{p}.
# $$

# In building a random forest, at
# each split in the tree, the algorithm is not even allowed to consider
# a majority of the available predictors. 
# 
# The reason for this is rather clever. Suppose that there is one very
# strong predictor in the data set, along with a number of other
# moderately strong predictors. Then in the collection of bagged
# variable importance random forest trees, most or all of the trees will
# use this strong predictor in the top split. Consequently, all of the
# bagged trees will look quite similar to each other. Hence the
# predictions from the bagged trees will be highly correlated.
# Unfortunately, averaging many highly correlated quantities does not
# lead to as large of a reduction in variance as averaging many
# uncorrelated quantities. In particular, this means that bagging will
# not lead to a substantial reduction in variance over a single tree in
# this setting.
# 
# 
# The algorithm described here can be applied to both classification and regression problems.
# 
# We will grow of forest of say $B$ trees.
# 1. For $b=1:B$
# 
#   * Draw a bootstrap sample from the training data organized in our $\boldsymbol{X}$ matrix.
# 
#   * We grow then a random forest tree $T_b$ based on the bootstrapped data by repeating the steps outlined till we reach the maximum node size is reached
# 
# 1. we select $m \le p$ variables at random from the $p$ predictors/features
# 
# 2. pick the best split point among the $m$ features using for example the CART algorithm and create a new node
# 
# 3. split the node into daughter nodes
# 
# 
# 
# 4. Output then the ensemble of trees $\{T_b\}_1^{B}$ and make predictions for either a regression type of problem or a classification type of problem.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
print(X_train.shape)
print(X_test.shape)
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression: {:.2f}".format(logreg.score(X_test,y_test)))
# Support vector machine
svm = SVC(gamma='auto', C=100)
svm.fit(X_train, y_train)
print("Test set accuracy with SVM: {:.2f}".format(svm.score(X_test,y_test)))
# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
deep_tree_clf.fit(X_train, y_train)
print("Test set accuracy with Decision Trees: {:.2f}".format(deep_tree_clf.score(X_test,y_test)))
#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Logistic Regression
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy Logistic Regression with scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))
# Support Vector Machine
svm.fit(X_train_scaled, y_train)
print("Test set accuracy SVM with scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))
# Decision Trees
deep_tree_clf.fit(X_train_scaled, y_train)
print("Test set accuracy with Decision Trees and scaled data: {:.2f}".format(deep_tree_clf.score(X_test_scaled,y_test)))


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
# Data set not specificied
#Instantiate the model with 500 trees and entropy as splitting criteria
Random_Forest_model = RandomForestClassifier(n_estimators=500,criterion="entropy")
Random_Forest_model.fit(X_train_scaled, y_train)
#Cross validation
accuracy = cross_validate(Random_Forest_model,X_test_scaled,y_test,cv=10)['test_score']
print(accuracy)
print("Test set accuracy with Random Forests and scaled data: {:.2f}".format(Random_Forest_model.score(X_test_scaled,y_test)))


import scikitplot as skplt
y_pred = Random_Forest_model.predict(X_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
y_probas = Random_Forest_model.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()


# Recall that the cumulative gains curve shows the percentage of the
# overall number of cases in a given category *gained* by targeting a
# percentage of the total number of cases.
# 
# Similarly, the receiver operating characteristic curve, or ROC curve,
# displays the diagnostic ability of a binary classifier system as its
# discrimination threshold is varied. It plots the true positive rate against the false positive rate.
# 
# 
# ### Compare  Bagging on Trees with Random Forests

# In[ ]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)


# In[ ]:


bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
np.sum(y_pred == y_pred_rf) / len(y_pred)


# ## Boosting, a Bird's Eye View
# 
# The basic idea is to combine weak classifiers in order to create a good
# classifier. With a weak classifier we often intend a classifier which
# produces results which are only slightly better than we would get by
# random guesses.
# 
# This is done by applying in an iterative way a weak (or a standard
# classifier like decision trees) to modify the data. In each iteration
# we emphasize those observations which are misclassified by weighting
# them with a factor.
# 
# 
# 
# Boosting is a way of fitting an additive expansion in a set of
# elementary basis functions like for example some simple polynomials.
# Assume for example that we have a function

# $$
# f_M(x) = \sum_{i=1}^M \beta_m b(x;\gamma_m),
# $$

# where $\beta_m$ are the expansion parameters to be determined in a
# minimization process and $b(x;\gamma_m)$ are some simple functions of
# the multivariable parameter $x$ which is characterized by the
# parameters $\gamma_m$.
# 
# As an example, consider the Sigmoid function we used in logistic
# regression. In that case, we can translate the function
# $b(x;\gamma_m)$ into the Sigmoid function

# $$
# \sigma(t) = \frac{1}{1+\exp{(-t)}},
# $$

# where $t=\gamma_0+\gamma_1 x$ and the parameters $\gamma_0$ and
# $\gamma_1$ were determined by the Logistic Regression fitting
# algorithm.
# 
# As another example, consider the cost function we defined for linear regression

# $$
# C(\boldsymbol{y},\boldsymbol{f}) = \frac{1}{n} \sum_{i=0}^{n-1}(y_i-f(x_i))^2.
# $$

# In this case the function $f(x)$ was replaced by the design matrix
# $\boldsymbol{X}$ and the unknown linear regression parameters $\boldsymbol{\beta}$,
# that is $\boldsymbol{f}=\boldsymbol{X}\boldsymbol{\beta}$. In linear regression we can 
# simply invert a matrix and obtain the parameters $\beta$ by

# $$
# \boldsymbol{\beta}=\left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}.
# $$

# In iterative fitting or additive modeling, we minimize the cost function with respect to the parameters $\beta_m$ and $\gamma_m$.
# 
# 
# ### Iterative Fitting, Regression and Squared-error Cost Function
# 
# The way we proceed is as follows (here we specialize to the squared-error cost function)
# 
# 1. Establish a cost function, here $\cal{C}(\boldsymbol{y},\boldsymbol{f}) = \frac{1}{n} \sum_{i=0}^{n-1}(y_i-f_M(x_i))^2$ with $f_M(x) = \sum_{i=1}^M \beta_m b(x;\gamma_m)$.
# 
# 2. Initialize with a guess $f_0(x)$. It could be one or even zero or some random numbers.
# 
# 3. For $m=1:M$
# 
# a. minimize $\sum_{i=0}^{n-1}(y_i-f_{m-1}(x_i)-\beta b(x;\gamma))^2$ wrt $\gamma$ and $\beta$
# 
# b. This gives the optimal values $\beta_m$ and $\gamma_m$
# 
# c. Determine then the new values $f_m(x)=f_{m-1}(x) +\beta_m b(x;\gamma_m)$
# 
# 
# We could use any of the algorithms we have discussed till now. If we
# use trees, $\gamma$ parameterizes the split variables and split points
# at the internal nodes, and the predictions at the terminal nodes.
# 
# 
# 
# To better understand what happens, let us develop the steps for the iterative fitting using the above squared error function.
# 
# For simplicity we assume also that our functions $b(x;\gamma)=1+\gamma x$. 
# 
# This means that for every iteration $m$, we need to optimize

# $$
# (\beta_m,\gamma_m) = \mathrm{argmin}_{\beta,\lambda}\hspace{0.1cm} \sum_{i=0}^{n-1}(y_i-f_{m-1}(x_i)-\beta b(x;\gamma))^2=\sum_{i=0}^{n-1}(y_i-f_{m-1}(x_i)-\beta(1+\gamma x_i))^2.
# $$

# We start our iteration by simply setting $f_0(x)=0$. 
# Taking the derivatives  with respect to $\beta$ and $\gamma$ we obtain

# $$
# \frac{\partial \cal{C}}{\partial \beta} = -2\sum_{i}(1+\gamma x_i)(y_i-\beta(1+\gamma x_i))=0,
# $$

# and

# $$
# \frac{\partial \cal{C}}{\partial \gamma} =-2\sum_{i}\beta x_i(y_i-\beta(1+\gamma x_i))=0.
# $$

# We can then rewrite these equations as (defining $\boldsymbol{w}=\boldsymbol{e}+\gamma \boldsymbol{x})$ with $\boldsymbol{e}$ being the unit vector)

# $$
# \gamma \boldsymbol{w}^T(\boldsymbol{y}-\beta\gamma \boldsymbol{w})=0,
# $$

# which gives us $\beta = \boldsymbol{w}^T\boldsymbol{y}/(\boldsymbol{w}^T\boldsymbol{w})$. Similarly we have

# $$
# \beta\gamma \boldsymbol{x}^T(\boldsymbol{y}-\beta(1+\gamma \boldsymbol{x}))=0,
# $$

# which leads to $\gamma =(\boldsymbol{x}^T\boldsymbol{y}-\beta\boldsymbol{x}^T\boldsymbol{e})/(\beta\boldsymbol{x}^T\boldsymbol{x})$.  Inserting
# for $\beta$ gives us an equation for $\gamma$. This is a non-linear equation in the unknown $\gamma$ and has to be solved numerically. 
# 
# The solution to these two equations gives us in turn $\beta_1$ and $\gamma_1$ leading to the new expression for $f_1(x)$ as
# $f_1(x) = \beta_1(1+\gamma_1x)$. Doing this $M$ times results in our final estimate for the function $f$. 
# 
# 
# 
# ### Iterative Fitting, Classification and AdaBoost
# 
# Let us consider a binary classification problem with two outcomes $y_i \in \{-1,1\}$ and $i=0,1,2,\dots,n-1$ as our set of
# observations. We define a classification function $G(x)$ which produces a prediction taking one or the other of the two values 
# $\{-1,1\}$.
# 
# The error rate of the training sample is then

# $$
# \mathrm{\overline{err}}=\frac{1}{n} \sum_{i=0}^{n-1} I(y_i\ne G(x_i)).
# $$

# The iterative procedure starts with defining a weak classifier whose
# error rate is barely better than random guessing.  The iterative
# procedure in boosting is to sequentially apply a  weak
# classification algorithm to repeatedly modified versions of the data
# producing a sequence of weak classifiers $G_m(x)$.
# 
# Here we will express our  function $f(x)$ in terms of $G(x)$. That is

# $$
# f_M(x) = \sum_{i=1}^M \beta_m b(x;\gamma_m),
# $$

# will be a function of

# $$
# G_M(x) = \mathrm{sign} \sum_{i=1}^M \alpha_m G_m(x).
# $$

# In our iterative procedure we define thus

# $$
# f_m(x) = f_{m-1}(x)+\beta_mG_m(x).
# $$

# The simplest possible cost function which leads (also simple from a computational point of view) to the AdaBoost algorithm is the
# exponential cost/loss function defined as

# $$
# C(\boldsymbol{y},\boldsymbol{f}) = \sum_{i=0}^{n-1}\exp{(-y_i(f_{m-1}(x_i)+\beta G(x_i))}.
# $$

# We optimize $\beta$ and $G$ for each value of $m=1:M$ as we did in the regression case.
# This is normally done in two steps. Let us however first rewrite the cost function as

# $$
# C(\boldsymbol{y},\boldsymbol{f}) = \sum_{i=0}^{n-1}w_i^{m}\exp{(-y_i\beta G(x_i))},
# $$

# where we have defined $w_i^m= \exp{(-y_if_{m-1}(x_i))}$.
# 
# 
# 
# First, for any $\beta > 0$, we optimize $G$ by setting

# $$
# G_m(x) = \mathrm{sign} \sum_{i=0}^{n-1} w_i^m I(y_i \ne G_(x_i)),
# $$

# which is the classifier that minimizes the weighted error rate in predicting $y$.
# 
# We can do this by rewriting

# $$
# \exp{-(\beta)}\sum_{y_i=G(x_i)}w_i^m+\exp{(\beta)}\sum_{y_i\ne G(x_i)}w_i^m,
# $$

# which can be rewritten as

# $$
# (\exp{(\beta)}-\exp{-(\beta)})\sum_{i=0}^{n-1}w_i^mI(y_i\ne G(x_i))+\exp{(-\beta)}\sum_{i=0}^{n-1}w_i^m=0,
# $$

# which leads to

# $$
# \beta_m = \frac{1}{2}\log{\frac{1-\mathrm{\overline{err}}}{\mathrm{\overline{err}}}},
# $$

# where we have redefined the error as

# $$
# \mathrm{\overline{err}}_m=\frac{1}{n}\frac{\sum_{i=0}^{n-1}w_i^mI(y_i\ne G(x_i)}{\sum_{i=0}^{n-1}w_i^m},
# $$

# which leads to an update of

# $$
# f_m(x) = f_{m-1}(x) +\beta_m G_m(x).
# $$

# This leads to the new weights

# $$
# w_i^{m+1} = w_i^m \exp{(-y_i\beta_m G_m(x_i))}
# $$

# ### Adaptive boosting: AdaBoost, Basic Algorithm
# 
# The algorithm here is rather straightforward. Assume that our weak
# classifier is a decision tree and we consider a binary set of outputs
# with $y_i \in \{-1,1\}$ and $i=0,1,2,\dots,n-1$ as our set of
# observations. Our design matrix is given in terms of the
# feature/predictor vectors
# $\boldsymbol{X}=[\boldsymbol{x}_0\boldsymbol{x}_1\dots\boldsymbol{x}_{p-1}]$. Finally, we define also a
# classifier determined by our data via a function $G(x)$. This function tells us how well we are able to classify our outputs/targets $\boldsymbol{y}$. 
# 
# We have already defined the misclassification error $\mathrm{err}$ as

# $$
# \mathrm{err}=\frac{1}{n}\sum_{i=0}^{n-1}I(y_i\ne G(x_i)),
# $$

# where the function $I()$ is one if we misclassify and zero if we classify correctly. 
# 
# 
# With the above definitions we are now ready to set up the algorithm for AdaBoost.
# The basic idea is to set up weights which will be used to scale the correctly classified and the misclassified cases.
# 1. We start by initializing all weights to $w_i = 1/n$, with $i=0,1,2,\dots n-1$. It is easy to see that we must have $\sum_{i=0}^{n-1}w_i = 1$.
# 
# 2. We rewrite the misclassification error as

# $$
# \mathrm{\overline{err}}_m=\frac{\sum_{i=0}^{n-1}w_i^m I(y_i\ne G(x_i))}{\sum_{i=0}^{n-1}w_i},
# $$

# 1. Then we start looping over all attempts at classifying, namely we start an iterative process for $m=1:M$, where $M$ is the final number of classifications. Our given classifier could for example be a plain decision tree.
# 
# a. Fit then a given classifier to the training set using the weights $w_i$.
# 
# b. Compute then $\mathrm{err}$ and figure out which events are classified properly and which are classified wrongly.
# 
# c. Define a quantity $\alpha_{m} = \log{(1-\mathrm{\overline{err}}_m)/\mathrm{\overline{err}}_m}$
# 
# d. Set the new weights to $w_i = w_i\times \exp{(\alpha_m I(y_i\ne G(x_i)}$.
# 
# 
# 5. Compute the new classifier $G(x)= \sum_{i=0}^{n-1}\alpha_m I(y_i\ne G(x_i)$.
# 
# For the iterations with $m \le 2$ the weights are modified
# individually at each steps. The observations which were misclassified
# at iteration $m-1$ have a weight which is larger than those which were
# classified properly. As this proceeds, the observations which were
# difficult to classifiy correctly are given a larger influence. Each
# new classification step $m$ is then forced to concentrate on those
# observations that are missed in the previous iterations.
# 
# 
# 
# 
# Using **Scikit-Learn** it is easy to apply the adaptive boosting algorithm, as done here.

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train_scaled, y_train)
y_pred = ada_clf.predict(X_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
y_probas = ada_clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas)
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
plt.show()


# ## Gradient boosting: Basics with Steepest Descent/Functional Gradient Descent
# 
# Gradient boosting is again a similar technique to Adaptive boosting,
# it combines so-called weak classifiers or regressors into a strong
# method via a series of iterations.
# 
# In order to understand the method, let us illustrate its basics by
# bringing back the essential steps in linear regression, where our cost
# function was the least squares function.
# 
# 
# We start again with our cost function $\cal{C}(\boldsymbol{y}m\boldsymbol{f})=\sum_{i=0}^{n-1}\cal{L}(y_i, f(x_i))$ where we want to minimize
# This means that for every iteration, we need to optimize

# $$
# (\hat{\boldsymbol{f}}) = \mathrm{argmin}_{\boldsymbol{f}}\hspace{0.1cm} \sum_{i=0}^{n-1}(y_i-f(x_i))^2.
# $$

# We define a real function $h_m(x)$ that defines our final function $f_M(x)$ as

# $$
# f_M(x) = \sum_{m=0}^M h_m(x).
# $$

# In the steepest decent approach we approximate $h_m(x) = -\rho_m g_m(x)$, where $\rho_m$ is a scalar and $g_m(x)$ the gradient defined as

# $$
# g_m(x_i) = \left[ \frac{\partial \cal{L}(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x_i)=f_{m-1}(x_i)}.
# $$

# With the new gradient we can update $f_m(x) = f_{m-1}(x) -\rho_m g_m(x)$. Using the above squared-error function we see that
# the gradient is $g_m(x_i) = -2(y_i-f(x_i))$.
# 
# Choosing $f_0(x)=0$ we obtain $g_m(x) = -2y_i$ and inserting this into the minimization problem for the cost function we have

# $$
# (\rho_1) = \mathrm{argmin}_{\rho}\hspace{0.1cm} \sum_{i=0}^{n-1}(y_i+2\rho y_i)^2.
# $$

# Optimizing with respect to $\rho$ we obtain (taking the derivative) that $\rho_1 = -1/2$. We have then that

# $$
# f_1(x) = f_{0}(x) -\rho_1 g_1(x)=-y_i.
# $$

# We can then proceed and compute

# $$
# g_2(x_i) = \left[ \frac{\partial \cal{L}(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x_i)=f_{1}(x_i)=y_i}=-4y_i,
# $$

# and find a new value for $\rho_2=-1/2$ and continue till we have reached $m=M$. We can modify the steepest descent method, or steepest boosting, by introducing what is called **gradient boosting**. 
# 
# 
# Steepest descent is however not much used, since it only optimizes $f$ at a fixed set of $n$ points,
# so we do not learn a function that can generalize. However, we can modify the algorithm by
# fitting a weak learner to approximate the negative gradient signal. 
# 
# Suppose we have a cost function $C(f)=\sum_{i=0}^{n-1}L(y_i, f(x_i))$ where $y_i$ is our target and $f(x_i)$ the function which is meant to model $y_i$. The above cost function could be our standard  squared-error  function

# $$
# C(\boldsymbol{y},\boldsymbol{f})=\sum_{i=0}^{n-1}(y_i-f(x_i))^2.
# $$

# The way we proceed in an iterative fashion is to
# 1. Initialize our estimate $f_0(x)$.
# 
# 2. For $m=1:M$, we
# 
# a. compute the negative gradient vector $\boldsymbol{u}_m = -\partial C(\boldsymbol{y},\boldsymbol{f})/\partial \boldsymbol{f}(x)$ at $f(x) = f_{m-1}(x)$;
# 
# b. fit the so-called base-learner to the negative gradient $h_m(u_m,x)$;
# 
# c. update the estimate $f_m(x) = f_{m-1}(x)+h_m(u_m,x)$;
# 
# 
# 4. The final estimate is then $f_M(x) = \sum_{m=1}^M h_m(u_m,x)$.
# 
# ## Gradient Boosting, Examples of Regression

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
from sklearn.metrics import mean_squared_error

n = 100
maxdegree = 6

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

for degree in range(1,maxdegree):
    model = GradientBoostingRegressor(max_depth=degree, n_estimators=100, learning_rate=1.0)  
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
    variance[degree] = np.mean( np.var(y_pred) )
    print('Max depth:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.xlim(1,maxdegree-1)
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
save_fig("gdregression")
plt.show()


# ## Gradient Boosting, Classification Example

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
import scikitplot as skplt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate

# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
print(X_train.shape)
print(X_test.shape)
#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

gd_clf = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=1.0)  
gd_clf.fit(X_train_scaled, y_train)
#Cross validation
accuracy = cross_validate(gd_clf,X_test_scaled,y_test,cv=10)['test_score']
print(accuracy)
print("Test set accuracy with Random Forests and scaled data: {:.2f}".format(gd_clf.score(X_test_scaled,y_test)))

import scikitplot as skplt
y_pred = gd_clf.predict(X_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
save_fig("gdclassiffierconfusion")
plt.show()
y_probas = gd_clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas)
save_fig("gdclassiffierroc")
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
save_fig("gdclassiffiercgain")
plt.show()


# ## XGBoost: Extreme Gradient Boosting
# 
# 
# [XGBoost](https://github.com/dmlc/xgboost) or Extreme Gradient
# Boosting, is an optimized distributed gradient boosting library
# designed to be highly efficient, flexible and portable. It implements
# machine learning algorithms under the Gradient Boosting
# framework. XGBoost provides a parallel tree boosting that solve many
# data science problems in a fast and accurate way. See the [article by Chen and Guestrin](https://arxiv.org/abs/1603.02754).
# 
# The authors design and build a highly scalable end-to-end tree
# boosting system. It has  a theoretically justified weighted quantile
# sketch for efficient proposal calculation. It introduces a novel sparsity-aware algorithm for parallel tree learning and an effective cache-aware block structure for out-of-core tree learning.
# 
# It is now the algorithm which wins essentially all ML competitions!!!
# 
# ## Regression Case

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
from sklearn.metrics import mean_squared_error

n = 100
maxdegree = 6

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

for degree in range(maxdegree):
    model =  xgb.XGBRegressor(objective ='reg:squarederror', colsaobjective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = degree, alpha = 10, n_estimators = 200)

    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred))**2 )
    variance[degree] = np.mean( np.var(y_pred) )
    print('Max depth:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.xlim(1,maxdegree-1)
plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()


# As you will see from the confusion matrix below, XGBoots does an excellent job on the Wisconsin cancer data and outperforms essentially all agorithms we have discussed till now.

# In[ ]:



import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scikitplot as skplt
import xgboost as xgb
# Load the data
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
print(X_train.shape)
print(X_test.shape)
#now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train_scaled,y_train)

y_test = xg_clf.predict(X_test_scaled)

print("Test set accuracy with Random Forests and scaled data: {:.2f}".format(xg_clf.score(X_test_scaled,y_test)))

import scikitplot as skplt
y_pred = xg_clf.predict(X_test_scaled)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
save_fig("xdclassiffierconfusion")
plt.show()
y_probas = xg_clf.predict_proba(X_test_scaled)
skplt.metrics.plot_roc(y_test, y_probas)
save_fig("xdclassiffierroc")
plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)
save_fig("gdclassiffiercgain")
plt.show()


xgb.plot_tree(xg_clf,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
save_fig("xgtree")
plt.show()

xgb.plot_importance(xg_clf)
plt.rcParams['figure.figsize'] = [5, 5]
save_fig("xgparams")
plt.show()

