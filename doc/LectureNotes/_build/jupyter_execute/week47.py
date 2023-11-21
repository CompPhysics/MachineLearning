#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html week47.do.txt --no_mako -->
# <!-- dom:TITLE: Week 47: From Decision Trees to Ensemble Methods, Random Forests and Boosting Methods   and Summary of Course -->

# # Week 47: From Decision Trees to Ensemble Methods, Random Forests and Boosting Methods   and Summary of Course
# **Morten Hjorth-Jensen**, Department of Physics, University of Oslo and Department of Physics and Astronomy and Facility for Rare Ion Beams, Michigan State University
# 
# Date: **November 20-24, 2023**

# ## Plan for week 47
# 
# **Active learning sessions on Tuesday and Wednesday.**
# 
#   * Work and Discussion of project 3
# 
#   * Last weekly exercise, course feedback, to be completed by Sunday November 26
# 
#   
# 
# **Material for the lecture on Thursday November 23, 2023.**
# 
#   * Thursday: Basics of decision trees, classification and regression algorithms and ensemble models 
# 
#   * Readings and Videos:
# 
#     * These lecture notes
# 
#     * [Video on Decision trees](https://www.youtube.com/watch?v=RmajweUFKvM&ab_channel=Simplilearn)
# 
#     * [Video on boosting methods by Hastie](https://www.youtube.com/watch?v=wPqtzj5VZus&ab_channel=H2O.ai)
# 
#     * [Video on AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA)
# 
#     * [Video on Gradient boost, part 1, parts 2-4 follow thereafter](https://www.youtube.com/watch?v=3CC4N4z3GJc)
# 
#     * Decision Trees: Geron's chapter 6 covers decision trees while ensemble models, voting and bagging are discussed in chapter 7. See also lecture from [STK-IN4300, lecture 7](https://www.uio.no/studier/emner/matnat/math/STK-IN4300/h20/slides/lecture_7.pdf). Chapter 9.2 of Hastie et al contains also a good discussion.

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

# ## More bagging
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

# ## Making your own Bootstrap: Changing the Level of the Decision Tree
# 
# Let us bring up our good old boostrap example from the linear regression lectures. We change the linerar regression algorithm with
# a decision tree wth different depths and perform a bootstrap aggregate (in this case we perform as many bootstraps as data points $n$).

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


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
 
mse_simpletree= np.mean( np.mean((y_test - simpleprediction)**2))
print("Simple tree:",mse_simpletree)
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

# ## Random Forest Algorithm
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
# 4. Output then the ensemble of trees $\{T_b\}_1^{B}$ and make predictions for either a regression type of problem or a classification type of problem.

# ## Random Forests Compared with other Methods on the Cancer Data

# In[2]:


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
#define methods
# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
# Support vector machine
svm = SVC(gamma='auto', C=100)
# Decision Trees
deep_tree_clf = DecisionTreeClassifier(max_depth=None)
#Scale the data
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

# ## Compare  Bagging on Trees with Random Forests

# In[3]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)


# In[4]:


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

# ## What is boosting? Additive Modelling/Iterative Fitting
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

# ## Iterative Fitting, Regression and Squared-error Cost Function
# 
# The way we proceed is as follows (here we specialize to the squared-error cost function)
# 
# 1. Establish a cost function, here ${\cal C}(\boldsymbol{y},\boldsymbol{f}) = \frac{1}{n} \sum_{i=0}^{n-1}(y_i-f_M(x_i))^2$ with $f_M(x) = \sum_{i=1}^M \beta_m b(x;\gamma_m)$.
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
# We could use any of the algorithms we have discussed till now. If we
# use trees, $\gamma$ parameterizes the split variables and split points
# at the internal nodes, and the predictions at the terminal nodes.

# ## Squared-Error Example and Iterative Fitting
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
# \frac{\partial {\cal C}}{\partial \beta} = -2\sum_{i}(1+\gamma x_i)(y_i-\beta(1+\gamma x_i))=0,
# $$

# and

# $$
# \frac{\partial {\cal C}}{\partial \gamma} =-2\sum_{i}\beta x_i(y_i-\beta(1+\gamma x_i))=0.
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

# ## Iterative Fitting, Classification and AdaBoost
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

# ## Adaptive Boosting, AdaBoost
# 
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

# ## Building up AdaBoost
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

# ## Adaptive boosting: AdaBoost, Basic Algorithm
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

# ## Basic Steps of AdaBoost
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
# 5. Compute the new classifier $G(x)= \sum_{i=0}^{n-1}\alpha_m I(y_i\ne G(x_i)$.
# 
# For the iterations with $m \le 2$ the weights are modified
# individually at each steps. The observations which were misclassified
# at iteration $m-1$ have a weight which is larger than those which were
# classified properly. As this proceeds, the observations which were
# difficult to classifiy correctly are given a larger influence. Each
# new classification step $m$ is then forced to concentrate on those
# observations that are missed in the previous iterations.

# ## AdaBoost Examples
# 
# Using **Scikit-Learn** it is easy to apply the adaptive boosting algorithm, as done here.

# In[5]:


from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.01, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()
y_probas = ada_clf.predict_proba(X_test)
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

# ## The Squared-Error again! Steepest Descent
# 
# We start again with our cost function ${\cal C}(\boldsymbol{y}m\boldsymbol{f})=\sum_{i=0}^{n-1}{\cal L}(y_i, f(x_i))$ where we want to minimize
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
# g_m(x_i) = \left[ \frac{\partial {\cal L}(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x_i)=f_{m-1}(x_i)}.
# $$

# With the new gradient we can update $f_m(x) = f_{m-1}(x) -\rho_m g_m(x)$. Using the above squared-error function we see that
# the gradient is $g_m(x_i) = -2(y_i-f(x_i))$.
# 
# Choosing $f_0(x)=0$ we obtain $g_m(x) = -2y_i$ and inserting this into the minimization problem for the cost function we have

# $$
# (\rho_1) = \mathrm{argmin}_{\rho}\hspace{0.1cm} \sum_{i=0}^{n-1}(y_i+2\rho y_i)^2.
# $$

# ## Steepest Descent Example
# 
# Optimizing with respect to $\rho$ we obtain (taking the derivative) that $\rho_1 = -1/2$. We have then that

# $$
# f_1(x) = f_{0}(x) -\rho_1 g_1(x)=-y_i.
# $$

# We can then proceed and compute

# $$
# g_2(x_i) = \left[ \frac{\partial {\cal L}(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x_i)=f_{1}(x_i)=y_i}=-4y_i,
# $$

# and find a new value for $\rho_2=-1/2$ and continue till we have reached $m=M$. We can modify the steepest descent method, or steepest boosting, by introducing what is called **gradient boosting**.

# ## Gradient Boosting, algorithm
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
# 4. The final estimate is then $f_M(x) = \sum_{m=1}^M h_m(u_m,x)$.

# ## Gradient Boosting, Examples of Regression

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

for degree in range(1,maxdegree):
    model = GradientBoostingRegressor(max_depth=degree, n_estimators=100, learning_rate=1.0)  
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
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

# In[7]:


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
print("Test set accuracy with Gradient boosting and scaled data: {:.2f}".format(gd_clf.score(X_test_scaled,y_test)))

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

# ## Regression Case

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
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

for degree in range(maxdegree):
    model =  xgb.XGBRegressor(objective ='reg:squarederror', colsaobjective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = degree, alpha = 10, n_estimators = 200)

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
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


# ## Xgboost on the Cancer Data
# 
# As you will see from the confusion matrix below, XGBoots does an excellent job on the Wisconsin cancer data and outperforms essentially all agorithms we have discussed till now.

# In[9]:



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

print("Test set accuracy with Gradient Boosting and scaled data: {:.2f}".format(xg_clf.score(X_test_scaled,y_test)))

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


# ## Summary of course

# ## What? Me worry? No final exam in this course!
# <!-- dom:FIGURE: [figures/exam1.jpeg, width=500 frac=0.6] -->
# <!-- begin figure -->
# 
# <img src="figures/exam1.jpeg" width="500"><p style="font-size: 0.9em"><i>Figure 1: </i></p>
# <!-- end figure -->

# ## What is the link between Artificial Intelligence and Machine Learning and some general Remarks
# 
# Artificial intelligence is built upon integrated machine learning
# algorithms as discussed in this course, which in turn are fundamentally rooted in optimization and
# statistical learning.
# 
# Can we have Artificial Intelligence without Machine Learning? See [this post for inspiration](https://www.linkedin.com/pulse/what-artificial-intelligence-without-machine-learning-claudia-pohlink).

# ## Going back to the beginning of the semester
# 
# Traditionally the field of machine learning has had its main focus on
# predictions and correlations.  These concepts outline in some sense
# the difference between machine learning and what is normally called
# Bayesian statistics or Bayesian inference.
# 
# In machine learning and prediction based tasks, we are often
# interested in developing algorithms that are capable of learning
# patterns from given data in an automated fashion, and then using these
# learned patterns to make predictions or assessments of newly given
# data. In many cases, our primary concern is the quality of the
# predictions or assessments, and we are less concerned with the
# underlying patterns that were learned in order to make these
# predictions.  This leads to what normally has been labeled as a
# frequentist approach.

# ## Not so sharp distinctions
# 
# You should keep in mind that the division between a traditional
# frequentist approach with focus on predictions and correlations only
# and a Bayesian approach with an emphasis on estimations and
# causations, is not that sharp. Machine learning can be frequentist
# with ensemble methods (EMB) as examples and Bayesian with Gaussian
# Processes as examples.
# 
# If one views ML from a statistical learning
# perspective, one is then equally interested in estimating errors as
# one is in finding correlations and making predictions. It is important
# to keep in mind that the frequentist and Bayesian approaches differ
# mainly in their interpretations of probability. In the frequentist
# world, we can only assign probabilities to repeated random
# phenomena. From the observations of these phenomena, we can infer the
# probability of occurrence of a specific event.  In Bayesian
# statistics, we assign probabilities to specific events and the
# probability represents the measure of belief/confidence for that
# event. The belief can be updated in the light of new evidence.

# ## Topics we have covered this year
# 
# The course has two central parts
# 
# 1. Statistical analysis and optimization of data
# 
# 2. Machine learning

# ## Statistical analysis and optimization of data
# 
# The following topics have been discussed:
# 1. Basic concepts, expectation values, variance, covariance, correlation functions and errors;
# 
# 2. Simpler models, binomial distribution, the Poisson distribution, simple and multivariate normal distributions;
# 
# 3. Central elements from linear algebra, matrix inversion and SVD
# 
# 4. Gradient methods for data optimization
# 
# 5. Estimation of errors using cross-validation, bootstrapping and jackknife methods;
# 
# 6. Practical optimization using Singular-value decomposition and least squares for parameterizing data.
# 
# 7. Principal Component Analysis to reduce the number of features.

# ## Machine learning
# 
# The following topics will be covered
# 1. Linear methods for regression and classification:
# 
# a. Ordinary Least Squares
# 
# b. Ridge regression
# 
# c. Lasso regression
# 
# d. Logistic regression
# 
# 5. Neural networks and deep learning:
# 
# a. Feed Forward Neural Networks
# 
# b. Convolutional Neural Networks
# 
# c. Recurrent Neural Networks
# 
# 4. Decisions trees and ensemble methods:
# 
# a. Decision trees
# 
# b. Bagging and voting
# 
# c. Random forests
# 
# d. Boosting and gradient boosting
# 
# 5. Support vector machines, not covered this year but included in notes
# 
# a. Binary classification and multiclass classification
# 
# b. Kernel methods
# 
# c. Regression

# ## Learning outcomes and overarching aims of this course
# 
# The course introduces a variety of central algorithms and methods
# essential for studies of data analysis and machine learning. The
# course is project based and through the various projects, normally
# three, you will be exposed to fundamental research problems
# in these fields, with the aim to reproduce state of the art scientific
# results. The students will learn to develop and structure large codes
# for studying these systems, get acquainted with computing facilities
# and learn to handle large scientific projects. A good scientific and
# ethical conduct is emphasized throughout the course. 
# 
# * Understand linear methods for regression and classification;
# 
# * Learn about neural network;
# 
# * Learn about bagging, boosting and trees
# 
# * Support vector machines, not covered
# 
# * Learn about basic data analysis;
# 
# * Be capable of extending the acquired knowledge to other systems and cases;
# 
# * Have an understanding of central algorithms used in data analysis and machine learning;
# 
# * Work on numerical projects to illustrate the theory. The projects play a central role and you are expected to know modern programming languages like Python or C++.

# ## Perspective on Machine Learning
# 
# 1. Rapidly emerging application area
# 
# 2. Experiment AND theory are evolving in many many fields. Still many low-hanging fruits.
# 
# 3. Requires education/retraining for more widespread adoption
# 
# 4. A lot of “word-of-mouth” development methods
# 
# Huge amounts of data sets require automation, classical analysis tools often inadequate. 
# High energy physics hit this wall in the 90’s.
# In 2009 single top quark production was determined via [Boosted decision trees, Bayesian
# Neural Networks, etc.](https://arxiv.org/pdf/0903.0850.pdf). Similarly, the search for Higgs was a statistical learning tour de force. See this link on [Kaggle.com](https://www.kaggle.com/c/higgs-boson).

# ## Machine Learning Research
# 
# Where to find recent results:
# 1. Conference proceedings, arXiv and blog posts!
# 
# 2. **NIPS**: [Neural Information Processing Systems](https://papers.nips.cc)
# 
# 3. **ICLR**: [International Conference on Learning Representations](https://openreview.net/group?id=ICLR.cc/2018/Conference#accepted-oral-papers)
# 
# 4. **ICML**: International Conference on Machine Learning
# 
# 5. [Journal of Machine Learning Research](http://www.jmlr.org/papers/v19/) 
# 
# 6. [Follow ML on ArXiv](https://arxiv.org/list/cs.LG/recent)

# ## Starting your Machine Learning Project
# 
# 1. Identify problem type: classification, regression
# 
# 2. Consider your data carefully
# 
# 3. Choose a simple model that fits 1. and 2.
# 
# 4. Consider your data carefully again! Think of data representation more carefully.
# 
# 5. Based on your results, feedback loop to earliest possible point

# ## Choose a Model and Algorithm
# 
# 1. Supervised?
# 
# 2. Start with the simplest model that fits your problem
# 
# 3. Start with minimal processing of data

# ## Preparing Your Data
# 
# 1. Shuffle your data
# 
# 2. Mean center your data
# 
#   * Why?
# 
# 3. Normalize the variance
# 
#   * Why?
# 
# 4. [Whitening](https://multivariatestatsjl.readthedocs.io/en/latest/whiten.html)
# 
#   * Decorrelates data
# 
#   * Can be hit or miss
# 
# 5. When to do train/test split?
# 
# Whitening is a decorrelation transformation that transforms a set of
# random variables into a set of new random variables with identity
# covariance (uncorrelated with unit variances).

# ## Which Activation and Weights to Choose in Neural Networks
# 
# 1. RELU? ELU?
# 
# 2. Sigmoid or Tanh?
# 
# 3. Set all weights to 0?
# 
#   * Terrible idea
# 
# 4. Set all weights to random values?
# 
#   * Small random values

# ## Optimization Methods and Hyperparameters
# 1. Stochastic gradient descent
# 
# a. Stochastic gradient descent + momentum
# 
# 2. State-of-the-art approaches:
# 
#   * RMSProp
# 
#   * Adam
# 
#   * and more
# 
# Which regularization and hyperparameters? $L_1$ or $L_2$, soft
# classifiers, depths of trees and many other. Need to explore a large
# set of hyperparameters and regularization methods.

# ## Resampling
# 
# When do we resample?
# 
# 1. [Bootstrap](https://www.cambridge.org/core/books/bootstrap-methods-and-their-application/ED2FD043579F27952363566DC09CBD6A)
# 
# 2. [Cross-validation](https://www.youtube.com/watch?v=fSytzGwwBVw&ab_channel=StatQuestwithJoshStarmer)
# 
# 3. Jackknife and many other

# ## Other courses on Data science and Machine Learning  at UiO
# 
# 1. [FYS5429 Advanced Machine Learning and Data Analysis for the Physical Sciences](https://www.uio.no/studier/emner/matnat/fys/FYS5429/index-eng.html)
# 
# 2. [FYS5419 Quantum Computing and Quantum Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS5419/index-eng.html)
# 
# 3. [STK2100 Machine learning and statistical methods for prediction and classification](http://www.uio.no/studier/emner/matnat/math/STK2100/index-eng.html). 
# 
# 4. [IN3050/IN4050 Introduction to Artificial Intelligence and Machine Learning](https://www.uio.no/studier/emner/matnat/ifi/IN3050/index-eng.html). Introductory course in machine learning and AI with an algorithmic approach. 
# 
# 5. [STK-INF3000/4000 Selected Topics in Data Science](http://www.uio.no/studier/emner/matnat/math/STK-INF3000/index-eng.html). The course provides insight into selected contemporary relevant topics within Data Science. 
# 
# 6. [IN4080 Natural Language Processing](https://www.uio.no/studier/emner/matnat/ifi/IN4080/index.html). Probabilistic and machine learning techniques applied to natural language processing. o [STK-IN4300 – Statistical learning methods in Data Science](https://www.uio.no/studier/emner/matnat/math/STK-IN4300/index-eng.html). An advanced introduction to statistical and machine learning. For students with a good mathematics and statistics background.
# 
# 7. [IN-STK5000  Adaptive Methods for Data-Based Decision Making](https://www.uio.no/studier/emner/matnat/ifi/IN-STK5000/index-eng.html). Methods for adaptive collection and processing of data based on machine learning techniques. 
# 
# 8. [IN5400/INF5860 – Machine Learning for Image Analysis](https://www.uio.no/studier/emner/matnat/ifi/IN5400/). An introduction to deep learning with particular emphasis on applications within Image analysis, but useful for other application areas too.
# 
# 9. [TEK5040 – Dyp læring for autonome systemer](https://www.uio.no/studier/emner/matnat/its/TEK5040/). The course addresses advanced algorithms and architectures for deep learning with neural networks. The course provides an introduction to how deep-learning techniques can be used in the construction of key parts of advanced autonomous systems that exist in physical environments and cyber environments.

# ## Additional courses of interest
# 
# 1. [STK4051 Computational Statistics](https://www.uio.no/studier/emner/matnat/math/STK4051/index-eng.html)
# 
# 2. [STK4021 Applied Bayesian Analysis and Numerical Methods](https://www.uio.no/studier/emner/matnat/math/STK4021/index-eng.html)

# ## What's the future like?
# 
# Based on multi-layer nonlinear neural networks, deep learning can
# learn directly from raw data, automatically extract and abstract
# features from layer to layer, and then achieve the goal of regression,
# classification, or ranking. Deep learning has made breakthroughs in
# computer vision, speech processing and natural language, and reached
# or even surpassed human level. The success of deep learning is mainly
# due to the three factors: big data, big model, and big computing.
# 
# In the past few decades, many different architectures of deep neural
# networks have been proposed, such as
# 1. Convolutional neural networks, which are mostly used in image and video data processing, and have also been applied to sequential data such as text processing;
# 
# 2. Recurrent neural networks, which can process sequential data of variable length and have been widely used in natural language understanding and speech processing;
# 
# 3. Encoder-decoder framework, which is mostly used for image or sequence generation, such as machine translation, text summarization, and image captioning.

# ## Types of Machine Learning, a repetition
# 
# The approaches to machine learning are many, but are often split into two main categories. 
# In *supervised learning* we know the answer to a problem,
# and let the computer deduce the logic behind it. On the other hand, *unsupervised learning*
# is a method for finding patterns and relationship in data sets without any prior knowledge of the system.
# Some authours also operate with a third category, namely *reinforcement learning*. This is a paradigm 
# of learning inspired by behavioural psychology, where learning is achieved by trial-and-error, 
# solely from rewards and punishment.
# 
# Another way to categorize machine learning tasks is to consider the desired output of a system.
# Some of the most common tasks are:
# 
#   * Classification: Outputs are divided into two or more classes. The goal is to   produce a model that assigns inputs into one of these classes. An example is to identify  digits based on pictures of hand-written ones. Classification is typically supervised learning.
# 
#   * Regression: Finding a functional relationship between an input data set and a reference data set.   The goal is to construct a function that maps input data to continuous output values.
# 
#   * Clustering: Data are divided into groups with certain common traits, without knowing the different groups beforehand.  It is thus a form of unsupervised learning.
# 
#   * Other unsupervised learning algortihms like **Boltzmann machines**

# ## Why Boltzmann machines?
# 
# What is known as restricted Boltzmann Machines (RMB) have received a lot of attention lately. 
# One of the major reasons is that they can be stacked layer-wise to build deep neural networks that capture complicated statistics.
# 
# The original RBMs had just one visible layer and a hidden layer, but recently so-called Gaussian-binary RBMs have gained quite some popularity in imaging since they are capable of modeling continuous data that are common to natural images. 
# 
# Furthermore, they have been used to solve complicated [quantum mechanical many-particle problems or classical statistical physics problems like the Ising and Potts classes of models](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.91.045002).

# ## Boltzmann Machines
# 
# Why use a generative model rather than the more well known discriminative deep neural networks (DNN)? 
# 
# * Discriminitave methods have several limitations: They are mainly supervised learning methods, thus requiring labeled data. And there are tasks they cannot accomplish, like drawing new examples from an unknown probability distribution.
# 
# * A generative model can learn to represent and sample from a probability distribution. The core idea is to learn a parametric model of the probability distribution from which the training data was drawn. As an example
# 
# a. A model for images could learn to draw new examples of cats and dogs, given a training dataset of images of cats and dogs.
# 
# b. Generate a sample of an ordered or disordered phase, having been given samples of such phases.
# 
# c. Model the trial function for [Monte Carlo calculations](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.91.045002).

# ## Some similarities and differences from DNNs
# 
# 1. Both use gradient-descent based learning procedures for minimizing cost functions
# 
# 2. Energy based models don't use backpropagation and automatic differentiation for computing gradients, instead turning to Markov Chain Monte Carlo methods.
# 
# 3. DNNs often have several hidden layers. A restricted Boltzmann machine has only one hidden layer, however several RBMs can be stacked to make up Deep Belief Networks, of which they constitute the building blocks.
# 
# History: The RBM was developed by amongst others [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton), called by some the "Godfather of Deep Learning", working with the University of Toronto and Google.

# ## Boltzmann machines (BM)
# 
# A BM is what we would call an undirected probabilistic graphical model
# with stochastic continuous or discrete units.
# 
# It is interpreted as a stochastic recurrent neural network where the
# state of each unit(neurons/nodes) depends on the units it is connected
# to. The weights in the network represent thus the strength of the
# interaction between various units/nodes.
# 
# It turns into a Hopfield network if we choose deterministic rather
# than stochastic units. In contrast to a Hopfield network, a BM is a
# so-called generative model. It allows us to generate new samples from
# the learned distribution.

# ## A standard BM setup
# 
# A standard BM network is divided into a set of observable and visible units $\hat{x}$ and a set of unknown hidden units/nodes $\hat{h}$.
# 
# Additionally there can be bias nodes for the hidden and visible layers. These biases are normally set to $1$.
# 
# BMs are stackable, meaning they cwe can train a BM which serves as input to another BM. We can construct deep networks for learning complex PDFs. The layers can be trained one after another, a feature which makes them popular in deep learning
# 
# However, they are often hard to train. This leads to the introduction of so-called restricted BMs, or RBMS.
# Here we take away all lateral connections between nodes in the visible layer as well as connections between nodes in the hidden layer. The network is illustrated in the figure below.

# ## The structure of the RBM network
# 
# <!-- dom:FIGURE: [figures/RBM.png, width=800 frac=1.0] -->
# <!-- begin figure -->
# 
# <img src="figures/RBM.png" width="800"><p style="font-size: 0.9em"><i>Figure 1: </i></p>
# <!-- end figure -->

# ## The network
# 
# **The network layers**:
# 1. A function $\mathbf{x}$ that represents the visible layer, a vector of $M$ elements (nodes). This layer represents both what the RBM might be given as training input, and what we want it to be able to reconstruct. This might for example be given by the pixels of an image or coefficients representing speech, or the coordinates of a quantum mechanical state function.
# 
# 2. The function $\mathbf{h}$ represents the hidden, or latent, layer. A vector of $N$ elements (nodes). Also called "feature detectors".

# ## Goals
# 
# The goal of the hidden layer is to increase the model's expressive
# power. We encode complex interactions between visible variables by
# introducing additional, hidden variables that interact with visible
# degrees of freedom in a simple manner, yet still reproduce the complex
# correlations between visible degrees in the data once marginalized
# over (integrated out).
# 
# **The network parameters, to be optimized/learned**:
# 1. $\mathbf{a}$ represents the visible bias, a vector of same length as $\mathbf{x}$.
# 
# 2. $\mathbf{b}$ represents the hidden bias, a vector of same lenght as $\mathbf{h}$.
# 
# 3. $W$ represents the interaction weights, a matrix of size $M\times N$.

# ## Joint distribution
# 
# The restricted Boltzmann machine is described by a Boltzmann distribution

# <!-- Equation labels as ordinary links -->
# <div id="_auto1"></div>
# 
# $$
# \begin{equation}
# 	P_{rbm}(\mathbf{x},\mathbf{h}) = \frac{1}{Z} e^{-\frac{1}{T_0}E(\mathbf{x},\mathbf{h})},
# \label{_auto1} \tag{1}
# \end{equation}
# $$

# where $Z$ is the normalization constant or partition function, defined as

# <!-- Equation labels as ordinary links -->
# <div id="_auto2"></div>
# 
# $$
# \begin{equation}
# 	Z = \int \int e^{-\frac{1}{T_0}E(\mathbf{x},\mathbf{h})} d\mathbf{x} d\mathbf{h}.
# \label{_auto2} \tag{2}
# \end{equation}
# $$

# It is common to ignore $T_0$ by setting it to one.

# ## Network Elements, the energy function
# 
# The function $E(\mathbf{x},\mathbf{h})$ gives the **energy** of a
# configuration (pair of vectors) $(\mathbf{x}, \mathbf{h})$. The lower
# the energy of a configuration, the higher the probability of it. This
# function also depends on the parameters $\mathbf{a}$, $\mathbf{b}$ and
# $W$. Thus, when we adjust them during the learning procedure, we are
# adjusting the energy function to best fit our problem.
# 
# An expression for the energy function is

# $$
# E(\hat{x},\hat{h}) = -\sum_{ia}^{NA}b_i^a \alpha_i^a(x_i)-\sum_{jd}^{MD}c_j^d \beta_j^d(h_j)-\sum_{ijad}^{NAMD}b_i^a \alpha_i^a(x_i)c_j^d \beta_j^d(h_j)w_{ij}^{ad}.
# $$

# Here $\beta_j^d(h_j)$ and $\alpha_i^a(x_j)$ are so-called transfer functions that map a given input value to a desired feature value. The labels $a$ and $d$ denote that there can be multiple transfer functions per variable. The first sum depends only on the visible units. The second on the hidden ones. **Note** that there is no connection between nodes in a layer.
# 
# The quantities $b$ and $c$ can be interpreted as the visible and hidden biases, respectively.
# 
# The connection between the nodes in the two layers is given by the weights $w_{ij}$.

# ## Defining different types of RBMs
# There are different variants of RBMs, and the differences lie in the types of visible and hidden units we choose as well as in the implementation of the energy function $E(\mathbf{x},\mathbf{h})$. 
# 
# **Binary-Binary RBM:**
# 
# RBMs were first developed using binary units in both the visible and hidden layer. The corresponding energy function is defined as follows:

# <!-- Equation labels as ordinary links -->
# <div id="_auto3"></div>
# 
# $$
# \begin{equation}
# 	E(\mathbf{x}, \mathbf{h}) = - \sum_i^M x_i a_i- \sum_j^N b_j h_j - \sum_{i,j}^{M,N} x_i w_{ij} h_j,
# \label{_auto3} \tag{3}
# \end{equation}
# $$

# where the binary values taken on by the nodes are most commonly 0 and 1.
# 
# **Gaussian-Binary RBM:**
# 
# Another varient is the RBM where the visible units are Gaussian while the hidden units remain binary:

# <!-- Equation labels as ordinary links -->
# <div id="_auto4"></div>
# 
# $$
# \begin{equation}
# 	E(\mathbf{x}, \mathbf{h}) = \sum_i^M \frac{(x_i - a_i)^2}{2\sigma_i^2} - \sum_j^N b_j h_j - \sum_{i,j}^{M,N} \frac{x_i w_{ij} h_j}{\sigma_i^2}. 
# \label{_auto4} \tag{4}
# \end{equation}
# $$

# ## More about RBMs
# 1. Useful when we model continuous data (i.e., we wish $\mathbf{x}$ to be continuous)
# 
# 2. Requires a smaller learning rate, since there's no upper bound to the value a component might take in the reconstruction
# 
# Other types of units include:
# 1. Softmax and multinomial units
# 
# 2. Gaussian visible and hidden units
# 
# 3. Binomial units
# 
# 4. Rectified linear units
# 
# To read more, see [Lectures on Boltzmann machines in Physics](https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/pub/notebook2/ipynb/notebook2.ipynb).

# ## Autoencoders: Overarching view
# 
# Autoencoders are artificial neural networks capable of learning
# efficient representations of the input data (these representations are called codings)  without
# any supervision (i.e., the training set is unlabeled). These codings
# typically have a much lower dimensionality than the input data, making
# autoencoders useful for dimensionality reduction. 
# 
# More importantly, autoencoders act as powerful feature detectors, and
# they can be used for unsupervised pretraining of deep neural networks.
# 
# Lastly, they are capable of randomly generating new data that looks
# very similar to the training data; this is called a generative
# model. For example, you could train an autoencoder on pictures of
# faces, and it would then be able to generate new faces.  Surprisingly,
# autoencoders work by simply learning to copy their inputs to their
# outputs. This may sound like a trivial task, but we will see that
# constraining the network in various ways can make it rather
# difficult. For example, you can limit the size of the internal
# representation, or you can add noise to the inputs and train the
# network to recover the original inputs. These constraints prevent the
# autoencoder from trivially copying the inputs directly to the outputs,
# which forces it to learn efficient ways of representing the data. In
# short, the codings are byproducts of the autoencoder’s attempt to
# learn the identity function under some constraints.
# 
# [Video on autoencoders](https://www.coursera.org/lecture/building-deep-learning-models-with-tensorflow/autoencoders-1U4L3)
# 
# See also A. Geron's textbook, chapter 15.

# ## Bayesian Machine Learning
# 
# This is an important topic if we aim at extracting a probability
# distribution. This gives us also a confidence interval and error
# estimates.
# 
# Bayesian machine learning allows us to encode our prior beliefs about
# what those models should look like, independent of what the data tells
# us. This is especially useful when we don’t have a ton of data to
# confidently learn our model.
# 
# [Video on Bayesian deep learning](https://www.youtube.com/watch?v=E1qhGw8QxqY&ab_channel=AndrewGordonWilson)
# 
# See also the [slides here](https://github.com/CompPhysics/MachineLearning/blob/master/doc/Articles/lec03.pdf).

# ## Reinforcement Learning
# 
# Reinforcement Learning (RL) is one of the most exciting fields of
# Machine Learning today, and also one of the oldest. It has been around
# since the 1950s, producing many interesting applications over the
# years.
# 
# It studies
# how agents take actions based on trial and error, so as to maximize
# some notion of cumulative reward in a dynamic system or
# environment. Due to its generality, the problem has also been studied
# in many other disciplines, such as game theory, control theory,
# operations research, information theory, multi-agent systems, swarm
# intelligence, statistics, and genetic algorithms.
# 
# In March 2016, AlphaGo, a computer program that plays the board game
# Go, beat Lee Sedol in a five-game match. This was the first time a
# computer Go program had beaten a 9-dan (highest rank) professional
# without handicaps. AlphaGo is based on deep convolutional neural
# networks and reinforcement learning. AlphaGo’s victory was a major
# milestone in artificial intelligence and it has also made
# reinforcement learning a hot research area in the field of machine
# learning.
# 
# [Lecture on Reinforcement Learning](https://www.youtube.com/watch?v=FgzM3zpZ55o&ab_channel=stanfordonline).
# 
# See also A. Geron's textbook, chapter 16.

# ## Transfer learning
# 
# The goal of transfer learning is to transfer the model or knowledge
# obtained from a source task to the target task, in order to resolve
# the issues of insufficient training data in the target task. The
# rationality of doing so lies in that usually the source and target
# tasks have inter-correlations, and therefore either the features,
# samples, or models in the source task might provide useful information
# for us to better solve the target task. Transfer learning is a hot
# research topic in recent years, with many problems still waiting to be studied.
# 
# [Lecture on transfer learning](https://www.ias.edu/video/machinelearning/2020/0331-SamoryKpotufe).

# ## Adversarial learning
# 
# The conventional deep generative model has a potential problem: the
# model tends to generate extreme instances to maximize the
# probabilistic likelihood, which will hurt its performance. Adversarial
# learning utilizes the adversarial behaviors (e.g., generating
# adversarial instances or training an adversarial model) to enhance the
# robustness of the model and improve the quality of the generated
# data. In recent years, one of the most promising unsupervised learning
# technologies, generative adversarial networks (GAN), has already been
# successfully applied to image, speech, and text.
# 
# [Lecture on adversial learning](https://www.youtube.com/watch?v=CIfsB_EYsVI&ab_channel=StanfordUniversitySchoolofEngineering).

# ## Dual learning
# 
# Dual learning is a new learning paradigm, the basic idea of which is
# to use the primal-dual structure between machine learning tasks to
# obtain effective feedback/regularization, and guide and strengthen the
# learning process, thus reducing the requirement of large-scale labeled
# data for deep learning. The idea of dual learning has been applied to
# many problems in machine learning, including machine translation,
# image style conversion, question answering and generation, image
# classification and generation, text classification and generation,
# image-to-text, and text-to-image.

# ## Distributed machine learning
# 
# Distributed computation will speed up machine learning algorithms,
# significantly improve their efficiency, and thus enlarge their
# application. When distributed meets machine learning, more than just
# implementing the machine learning algorithms in parallel is required.

# ## Meta learning
# 
# Meta learning is an emerging research direction in machine
# learning. Roughly speaking, meta learning concerns learning how to
# learn, and focuses on the understanding and adaptation of the learning
# itself, instead of just completing a specific learning task. That is,
# a meta learner needs to be able to evaluate its own learning methods
# and adjust its own learning methods according to specific learning
# tasks.

# ## The Challenges Facing Machine Learning
# 
# While there has been much progress in machine learning, there are also challenges.
# 
# For example, the mainstream machine learning technologies are
# black-box approaches, making us concerned about their potential
# risks. To tackle this challenge, we may want to make machine learning
# more explainable and controllable. As another example, the
# computational complexity of machine learning algorithms is usually
# very high and we may want to invent lightweight algorithms or
# implementations. Furthermore, in many domains such as physics,
# chemistry, biology, and social sciences, people usually seek elegantly
# simple equations (e.g., the Schrödinger equation) to uncover the
# underlying laws behind various phenomena. In the field of machine
# learning, can we reveal simple laws instead of designing more complex
# models for data fitting? Although there are many challenges, we are
# still very optimistic about the future of machine learning. As we look
# forward to the future, here are what we think the research hotspots in
# the next ten years will be.
# 
# See the article on [Discovery of Physics From Data: Universal Laws and Discrepancies](https://www.frontiersin.org/articles/10.3389/frai.2020.00025/full)

# ## Explainable machine learning
# 
# Machine learning, especially deep learning, evolves rapidly. The
# ability gap between machine and human on many complex cognitive tasks
# becomes narrower and narrower. However, we are still in the very early
# stage in terms of explaining why those effective models work and how
# they work.
# 
# **What is missing: the gap between correlation and causation**. Standard Machine Learning is based on what e have called a frequentist approach. 
# 
# Most
# machine learning techniques, especially the statistical ones, depend
# highly on correlations in data sets to make predictions and analyses. In
# contrast, rational humans tend to reply on clear and trustworthy
# causality relations obtained via logical reasoning on real and clear
# facts. It is one of the core goals of explainable machine learning to
# transition from solving problems by data correlation to solving
# problems by logical reasoning.
# 
# **Bayesian Machine Learning is one of the exciting research directions in this field**.

# ## Scientific Machine Learning
# 
# An important and emerging field is what has been dubbed as scientific ML, see the article by Deiana et al [Applications and Techniques for Fast Machine Learning in Science, arXiv:2110.13041](https://arxiv.org/abs/2110.13041)
# 
# The authors discuss applications and techniques for fast machine
# learning (ML) in science - the concept of integrating power ML
# methods into the real-time experimental data processing loop to
# accelerate scientific discovery. The report covers three main areas
# 
# 1. applications for fast ML across a number of scientific domains;
# 
# 2. techniques for training and implementing performant and resource-efficient ML algorithms;
# 
# 3. and computing architectures, platforms, and technologies for deploying these algorithms.

# ## Quantum machine learning
# 
# Quantum machine learning is an emerging interdisciplinary research
# area at the intersection of quantum computing and machine learning.
# 
# Quantum computers use effects such as quantum coherence and quantum
# entanglement to process information, which is fundamentally different
# from classical computers. Quantum algorithms have surpassed the best
# classical algorithms in several problems (e.g., searching for an
# unsorted database, inverting a sparse matrix), which we call quantum
# acceleration.
# 
# When quantum computing meets machine learning, it can be a mutually
# beneficial and reinforcing process, as it allows us to take advantage
# of quantum computing to improve the performance of classical machine
# learning algorithms. In addition, we can also use the machine learning
# algorithms (on classic computers) to analyze and improve quantum
# computing systems.
# 
# [Lecture on Quantum ML](https://www.youtube.com/watch?v=Xh9pUu3-WxM&ab_channel=InstituteforPure%26AppliedMathematics%28IPAM%29).
# 
# [Read interview with Maria Schuld on her work on Quantum Machine Learning](https://physics.aps.org/articles/v13/179?utm_campaign=weekly&utm_medium=email&utm_source=emailalert). See also [her recent textbook](https://www.springer.com/gp/book/9783319964232).

# ## Quantum machine learning algorithms based on linear algebra
# 
# Many quantum machine learning algorithms are based on variants of
# quantum algorithms for solving linear equations, which can efficiently
# solve N-variable linear equations with complexity of O(log2 N) under
# certain conditions. The quantum matrix inversion algorithm can
# accelerate many machine learning methods, such as least square linear
# regression, least square version of support vector machine, Gaussian
# process, and more. The training of these algorithms can be simplified
# to solve linear equations. The key bottleneck of this type of quantum
# machine learning algorithms is data input—that is, how to initialize
# the quantum system with the entire data set. Although efficient
# data-input algorithms exist for certain situations, how to efficiently
# input data into a quantum system is as yet unknown for most cases.

# ## Quantum reinforcement learning
# 
# In quantum reinforcement learning, a quantum agent interacts with the
# classical environment to obtain rewards from the environment, so as to
# adjust and improve its behavioral strategies. In some cases, it
# achieves quantum acceleration by the quantum processing capabilities
# of the agent or the possibility of exploring the environment through
# quantum superposition. Such algorithms have been proposed in
# superconducting circuits and systems of trapped ions.

# ## Quantum deep learning
# 
# Dedicated quantum information processors, such as quantum annealers
# and programmable photonic circuits, are well suited for building deep
# quantum networks. The simplest deep quantum network is the Boltzmann
# machine. The classical Boltzmann machine consists of bits with tunable
# interactions and is trained by adjusting the interaction of these bits
# so that the distribution of its expression conforms to the statistics
# of the data. To quantize the Boltzmann machine, the neural network can
# simply be represented as a set of interacting quantum spins that
# correspond to an adjustable Ising model. Then, by initializing the
# input neurons in the Boltzmann machine to a fixed state and allowing
# the system to heat up, we can read out the output qubits to get the
# result.

# ## Social machine learning
# 
# Machine learning aims to imitate how humans
# learn. While we have developed successful machine learning algorithms,
# until now we have ignored one important fact: humans are social. Each
# of us is one part of the total society and it is difficult for us to
# live, learn, and improve ourselves, alone and isolated. Therefore, we
# should design machines with social properties. Can we let machines
# evolve by imitating human society so as to achieve more effective,
# intelligent, interpretable “social machine learning”?
# 
# And much more.

# ## The last words?
# 
# Early computer scientist Alan Kay said, **The best way to predict the
# future is to create it**. Therefore, all machine learning
# practitioners, whether scholars or engineers, professors or students,
# need to work together to advance these important research
# topics. Together, we will not just predict the future, but create it.

# ## AI/ML and some statements you may have heard (and what do they mean?)
# 
# 1. Fei-Fei Li on ImageNet: **map out the entire world of objects** ([The data that transformed AI research](https://cacm.acm.org/news/219702-the-data-that-transformed-ai-research-and-possibly-the-world/fulltext))
# 
# 2. Russell and Norvig in their popular textbook: **relevant to any intellectual task; it is truly a universal field** ([Artificial Intelligence, A modern approach](http://aima.cs.berkeley.edu/))
# 
# 3. Woody Bledsoe puts it more bluntly: **in the long run, AI is the only science** (quoted in Pamilla McCorduck, [Machines who think](https://www.pamelamccorduck.com/machines-who-think))
# 
# If you wish to have a critical read on AI/ML from a societal point of view, see [Kate Crawford's recent text Atlas of AI](https://www.katecrawford.net/)
# 
# **Here: with AI/ML we intend a collection of machine learning methods with an emphasis on statistical learning and data analysis**

# ## Best wishes to you all and thanks so much for your heroic efforts this semester
# 
# <!-- dom:FIGURE: [figures/Nebbdyr2.png, width=500 frac=0.6] -->
# <!-- begin figure -->
# 
# <img src="figures/Nebbdyr2.png" width="500"><p style="font-size: 0.9em"><i>Figure 1: </i></p>
# <!-- end figure -->
