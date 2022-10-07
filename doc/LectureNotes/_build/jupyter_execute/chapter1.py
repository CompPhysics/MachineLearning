#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html chapter1.do.txt  -->

# # Linear Regression

# ## Introduction
# 
# Our emphasis throughout this series of lectures is on understanding
# the mathematical aspects of different algorithms used in the fields of
# data analysis and machine learning.
# 
# However, where possible we will emphasize the importance of using
# available software. We start thus with a hands-on and top-down
# approach to machine learning. The aim is thus to start with relevant
# data or data we have produced and use these to introduce statistical
# data analysis concepts and machine learning algorithms before we delve
# into the algorithms themselves. The examples we will use in the
# beginning, start with simple polynomials with random noise added. We
# will use the Python software package
# [Scikit-Learn](http://scikit-learn.org/stable/) and introduce various
# machine learning algorithms to make fits of the data and
# predictions. We move thereafter to more interesting cases such as data
# from say experiments (below we will look at experimental nuclear
# binding energies as an example).  These are examples where we can
# easily set up the data and then use machine learning algorithms
# included in for example **Scikit-Learn**.
# 
# These examples will serve us the purpose of getting
# started. Furthermore, they allow us to catch more than two birds with
# a stone. They will allow us to bring in some programming specific
# topics and tools as well as showing the power of various Python
# libraries for machine learning and statistical data analysis.
# 
# Here, we will mainly focus on two specific Python packages for Machine
# Learning, Scikit-Learn and Tensorflow (see below for links etc).
# Moreover, the examples we introduce will serve as inputs to many of
# our discussions later, as well as allowing you to set up models and
# produce your own data and get started with programming.

# ## What is Machine Learning?
# 
# Statistics, data science and machine learning form important fields of
# research in modern science.  They describe how to learn and make
# predictions from data, as well as allowing us to extract important
# correlations about physical process and the underlying laws of motion
# in large data sets. The latter, big data sets, appear frequently in
# essentially all disciplines, from the traditional Science, Technology,
# Mathematics and Engineering fields to Life Science, Law, education
# research, the Humanities and the Social Sciences. 
# 
# It has become more
# and more common to see research projects on big data in for example
# the Social Sciences where extracting patterns from complicated survey
# data is one of many research directions.  Having a solid grasp of data
# analysis and machine learning is thus becoming central to scientific
# computing in many fields, and competences and skills within the fields
# of machine learning and scientific computing are nowadays strongly
# requested by many potential employers. The latter cannot be
# overstated, familiarity with machine learning has almost become a
# prerequisite for many of the most exciting employment opportunities,
# whether they are in bioinformatics, life science, physics or finance,
# in the private or the public sector. This author has had several
# students or met students who have been hired recently based on their
# skills and competences in scientific computing and data science, often
# with marginal knowledge of machine learning.
# 
# Machine learning is a subfield of computer science, and is closely
# related to computational statistics.  It evolved from the study of
# pattern recognition in artificial intelligence (AI) research, and has
# made contributions to AI tasks like computer vision, natural language
# processing and speech recognition. Many of the methods we will study are also 
# strongly rooted in basic mathematics and physics research. 
# 
# Ideally, machine learning represents the science of giving computers
# the ability to learn without being explicitly programmed.  The idea is
# that there exist generic algorithms which can be used to find patterns
# in a broad class of data sets without having to write code
# specifically for each problem. The algorithm will build its own logic
# based on the data.  You should however always keep in mind that
# machines and algorithms are to a large extent developed by humans. The
# insights and knowledge we have about a specific system, play a central
# role when we develop a specific machine learning algorithm. 
# 
# Machine learning is an extremely rich field, in spite of its young
# age. The increases we have seen during the last three decades in
# computational capabilities have been followed by developments of
# methods and techniques for analyzing and handling large data sets,
# relying heavily on statistics, computer science and mathematics.  The
# field is rather new and developing rapidly. Popular software packages
# written in Python for machine learning like
# [Scikit-learn](http://scikit-learn.org/stable/),
# [Tensorflow](https://www.tensorflow.org/),
# [PyTorch](http://pytorch.org/) and [Keras](https://keras.io/), all
# freely available at their respective GitHub sites, encompass
# communities of developers in the thousands or more. And the number of
# code developers and contributors keeps increasing. Not all the
# algorithms and methods can be given a rigorous mathematical
# justification, opening up thereby large rooms for experimenting and
# trial and error and thereby exciting new developments.  However, a
# solid command of linear algebra, multivariate theory, probability
# theory, statistical data analysis, understanding errors and Monte
# Carlo methods are central elements in a proper understanding of many
# of algorithms and methods we will discuss.
# 
# The approaches to machine learning are many, but are often split into
# two main categories.  In *supervised learning* we know the answer to a
# problem, and let the computer deduce the logic behind it. On the other
# hand, *unsupervised learning* is a method for finding patterns and
# relationship in data sets without any prior knowledge of the system.
# Some authors also operate with a third category, namely
# *reinforcement learning*. This is a paradigm of learning inspired by
# behavioral psychology, where learning is achieved by trial-and-error,
# solely from rewards and punishment.
# 
# Another way to categorize machine learning tasks is to consider the
# desired output of a system.  Some of the most common tasks are:
# 
#   * Classification: Outputs are divided into two or more classes. The goal is to   produce a model that assigns inputs into one of these classes. An example is to identify  digits based on pictures of hand-written ones. Classification is typically supervised learning.
# 
#   * Regression: Finding a functional relationship between an input data set and a reference data set.   The goal is to construct a function that maps input data to continuous output values.
# 
#   * Clustering: Data are divided into groups with certain common traits, without knowing the different groups beforehand.  It is thus a form of unsupervised learning.
# 
# The methods we cover have three main topics in common, irrespective of
# whether we deal with supervised or unsupervised learning.
# * The first ingredient is normally our data set (which can be subdivided into training, validation  and test data). Many find the most difficult part of using Machine Learning to be the set up of your data in a meaningful way. 
# 
# * The second item is a model which is normally a function of some parameters.  The model reflects our knowledge of the system (or lack thereof). As an example, if we know that our data show a behavior similar to what would be predicted by a polynomial, fitting our data to a polynomial of some degree would then determin our model. 
# 
# * The last ingredient is a so-called **cost/loss** function (or error or risk function) which allows us to present an estimate on how good our model is in reproducing the data it is supposed to train.  
# 
# At the heart of basically all Machine Learning algorithms we will encounter so-called minimization or optimization algorithms. A large family of such methods are so-called **gradient methods**.

# ### A Frequentist approach to data analysis
# 
# When you hear phrases like **predictions and estimations** and
# **correlations and causations**, what do you think of?  May be you think
# of the difference between classifying new data points and generating
# new data points.
# Or perhaps you consider that correlations represent some kind of symmetric statements like
# if $A$ is correlated with $B$, then $B$ is correlated with
# $A$. Causation on the other hand is directional, that is if $A$ causes $B$, $B$ does not
# necessarily cause $A$.
# 
# These concepts are in some sense the difference between machine
# learning and statistics. In machine learning and prediction based
# tasks, we are often interested in developing algorithms that are
# capable of learning patterns from given data in an automated fashion,
# and then using these learned patterns to make predictions or
# assessments of newly given data. In many cases, our primary concern
# is the quality of the predictions or assessments, and we are less
# concerned about the underlying patterns that were learned in order
# to make these predictions.
# 
# In machine learning we normally use [a so-called frequentist approach](https://en.wikipedia.org/wiki/Frequentist_inference),
# where the aim is to make predictions and find correlations. We focus
# less on for example extracting a probability distribution function (PDF). The PDF can be
# used in turn to make estimations and find causations such as given $A$
# what is the likelihood of finding $B$.

# ### What is a good model?
# 
# In science and engineering we often end up in situations where we want to infer (or learn) a
# quantitative model $M$ for a given set of sample points $\boldsymbol{X} \in [x_1, x_2,\dots x_N]$.
# 
# As we will see repeatedly in these lectures, we could try to fit these data points to a model given by a
# straight line, or if we wish to be more sophisticated to a more complex
# function.
# 
# The reason for inferring such a model is that it
# serves many useful purposes. On the one hand, the model can reveal information
# encoded in the data or underlying mechanisms from which the data were generated. For instance, we could discover important
# correlations that relate interesting physics interpretations.
# 
# In addition, it can simplify the representation of the given data set and help
# us in making predictions about  future data samples.
# 
# A first important consideration to keep in mind is that inferring the *correct* model
# for a given data set is an elusive, if not impossible, task. The fundamental difficulty
# is that if we are not specific about what we mean by a *correct* model, there
# could easily be many different models that fit the given data set *equally well*.
# 
# The central question is this: what leads us to say that a model is correct or
# optimal for a given data set? To make the model inference problem well posed, i.e.,
# to guarantee that there is a unique optimal model for the given data, we need to
# impose additional assumptions or restrictions on the class of models considered. To
# this end, we should not be looking for just any model that can describe the data.
# Instead, we should look for a **model** $M$ that is the best among a restricted class
# of models. In addition, to make the model inference problem computationally
# tractable, we need to specify how restricted the class of models needs to be. A
# common strategy is to start 
# with the simplest possible class of models that is just necessary to describe the data
# or solve the problem at hand. More precisely, the model class should be rich enough
# to contain at least one model that can fit the data to a desired accuracy and yet be
# restricted enough that it is relatively simple to find the best model for the given data.
# 
# Thus, the most popular strategy is to start from the
# simplest class of models and increase the complexity of the models only when the
# simpler models become inadequate. For instance, if we work with a regression problem to fit a set of sample points, one
# may first try the simplest class of models, namely linear models, followed obviously by more complex models.
# 
# How to evaluate which model fits best the data is something we will come back to over and over again in these sets of lectures.

# ## Simple linear regression model using **scikit-learn**
# 
# We start with perhaps our simplest possible example, using
# **Scikit-Learn** to perform linear regression analysis on a data set
# produced by us.
# 
# What follows is a simple Python code where we have defined a function
# $y$ in terms of the variable $x$. Both are defined as vectors with  $100$ entries. 
# The numbers in the vector $\boldsymbol{x}$ are given
# by random numbers generated with a uniform distribution with entries
# $x_i \in [0,1]$ (more about probability distribution functions
# later). These values are then used to define a function $y(x)$
# (tabulated again as a vector) with a linear dependence on $x$ plus a
# random noise added via the normal distribution.
# 
# The Numpy functions are imported used the **import numpy as np**
# statement and the random number generator for the uniform distribution
# is called using the function **np.random.rand()**, where we specificy
# that we want $100$ random variables.  Using Numpy we define
# automatically an array with the specified number of elements, $100$ in
# our case.  With the Numpy function **randn()** we can compute random
# numbers with the normal distribution (mean value $\mu$ equal to zero and
# variance $\sigma^2$ set to one) and produce the values of $y$ assuming a linear
# dependence as function of $x$

# $$
# y = 2x+N(0,1),
# $$

# where $N(0,1)$ represents random numbers generated by the normal
# distribution.  From **Scikit-Learn** we import then the
# **LinearRegression** functionality and make a prediction $\tilde{y} =
# \alpha + \beta x$ using the function **fit(x,y)**. We call the set of
# data $(\boldsymbol{x},\boldsymbol{y})$ for our training data. The Python package
# **scikit-learn** has also a functionality which extracts the above
# fitting parameters $\alpha$ and $\beta$ (see below). Later we will
# distinguish between training data and test data.
# 
# For plotting we use the Python package
# [matplotlib](https://matplotlib.org/) which produces publication
# quality figures. Feel free to explore the extensive
# [gallery](https://matplotlib.org/gallery/index.html) of examples. In
# this example we plot our original values of $x$ and $y$ as well as the
# prediction **ypredict** ($\tilde{y}$), which attempts at fitting our
# data with a straight line.  Note also that **Scikit-Learn** requires a
# matrix as input for the input values $x$ and $y$. In the above code we
# have solved this by declaring $x$ and $y$ as arrays of dimension
# $n\times 1$.
# 
# In the code here we have also made a new array for $x\in [0,1]$. Our
# prediction is computed for these values, meaning that they were not
# included in the data set used to *train* (or fit) the model.
# This is a recurrring theme in machine learning and data analysis. We would like to train a model on a specific given data set.
# Thereafter we wish to apply it to data which were not included in the training. Below we will encounter this again in the so-called *train-validate-test* spliting. We will typically split our data into different sets, oen for training, one for validation and finally, our data from the untouched test vault!
# 
# The Python code follows here.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Importing various packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)
y = 2*x+np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
# This is our new x-array to which we test our model
xnew = np.array([[0],[1]])
ypredict = linreg.predict(xnew)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,1.0,0, 5.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()


# This example serves several aims. It allows us to demonstrate several
# aspects of data analysis and later machine learning algorithms. The
# immediate visualization shows that our linear fit is not
# impressive. It goes through the data points, but there are many
# outliers which are not reproduced by our linear regression.  We could
# now play around with this small program and change for example the
# factor in front of $x$ and the normal distribution.  Try to change the
# function $y$ to

# $$
# y = 10x+0.01 \times N(0,1),
# $$

# where $x$ is defined as before.  Does the fit look better? Indeed, by
# reducing the role of the noise given by the normal distribution we see immediately that
# our linear prediction seemingly reproduces better the training
# set. However, this testing 'by the eye' is obviously not satisfactory in the
# long run. Here we have only defined the training data and our model, and 
# have not discussed a more rigorous approach to the **cost** function.
# 
# We need more rigorous criteria in defining whether we have succeeded or
# not in modeling our training data.  You will be surprised to see that
# many scientists seldomly venture beyond this 'by the eye' approach. A
# standard approach for the *cost* function is the so-called $\chi^2$
# function (a variant of the mean-squared error (MSE))

# $$
# \chi^2 = \frac{1}{n}
# \sum_{i=0}^{n-1}\frac{(y_i-\tilde{y}_i)^2}{\sigma_i^2},
# $$

# where $\sigma_i^2$ is the variance (to be defined later) of the entry
# $y_i$.  We may not know the explicit value of $\sigma_i^2$, it serves
# however the aim of scaling the equations and make the cost function
# dimensionless.  
# 
# Minimizing the cost function is a central aspect of
# our discussions to come. Finding its minima as function of the model
# parameters ($\alpha$ and $\beta$ in our case) will be a recurring
# theme in these series of lectures. Essentially all machine learning
# algorithms we will discuss center around the minimization of the
# chosen cost function. This depends in turn on our specific
# model for describing the data, a typical situation in supervised
# learning. Automatizing the search for the minima of the cost function is a
# central ingredient in all algorithms. Typical methods which are
# employed are various variants of **gradient** methods. These will be
# discussed in more detail later. Again, you'll be surprised to hear that
# many practitioners minimize the above function ''by the eye', popularly dubbed as 
# 'chi by the eye'. That is, change a parameter and see (visually and numerically) that 
# the  $\chi^2$ function becomes smaller. 
# 
# There are many ways to define the cost function. A simpler approach is to look at the relative difference between the training data and the predicted data, that is we define 
# the relative error (why would we prefer the MSE instead of the relative error?) as

# $$
# \epsilon_{\mathrm{relative}}= \frac{\vert \boldsymbol{y} -\boldsymbol{\tilde{y}}\vert}{\vert \boldsymbol{y}\vert}.
# $$

# The squared cost function results in an arithmetic mean-unbiased
# estimator, and the absolute-value cost function results in a
# median-unbiased estimator (in the one-dimensional case, and a
# geometric median-unbiased estimator for the multi-dimensional
# case). The squared cost function has the disadvantage that it has the tendency
# to be dominated by outliers.
# 
# We can modify easily the above Python code and plot the relative error instead

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Number of data points
n = 100
x = np.random.rand(100,1)
y = 5*x+0.01*np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
ypredict = linreg.predict(x)

plt.plot(x, np.abs(ypredict-y)/abs(y), "ro")
plt.axis([0,1.0,0.0, 0.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{\mathrm{relative}}$')
plt.title(r'Relative error')
plt.show()


# Depending on the parameter in front of the normal distribution, we may
# have a small or larger relative error. Try to play around with
# different training data sets and study (graphically) the value of the
# relative error.
# 
# As mentioned above, **Scikit-Learn** has an impressive functionality.
# We can for example extract the values of $\alpha$ and $\beta$ and
# their error estimates, or the variance and standard deviation and many
# other properties from the statistical data analysis. 
# 
# Here we show an
# example of the functionality of **Scikit-Learn**.

# In[3]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

x = np.random.rand(100,1)
y = 2.0+ 5*x+0.5*np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
ypredict = linreg.predict(x)
print('The intercept alpha: \n', linreg.intercept_)
print('Coefficient beta : \n', linreg.coef_)
# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(y, ypredict))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(y, ypredict))
# Mean squared log error                                                        
print('Mean squared log error: %.2f' % mean_squared_log_error(y, ypredict) )
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(y, ypredict))
plt.plot(x, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0.0,1.0,1.5, 7.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression fit ')
plt.show()


# The function **coef** gives us the parameter $\beta$ of our fit while **intercept** yields 
# $\alpha$. Depending on the constant in front of the normal distribution, we get values near or far from $alpha =2$ and $\beta =5$. Try to play around with different parameters in front of the normal distribution. The function **meansquarederror** gives us the mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss defined as

# $$
# MSE(\boldsymbol{y},\boldsymbol{\tilde{y}}) = \frac{1}{n}
# \sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2,
# $$

# The smaller the value, the better the fit. Ideally we would like to
# have an MSE equal zero.  The attentive reader has probably recognized
# this function as being similar to the $\chi^2$ function defined above.
# 
# The **r2score** function computes $R^2$, the coefficient of
# determination. It provides a measure of how well future samples are
# likely to be predicted by the model. Best possible score is 1.0 and it
# can be negative (because the model can be arbitrarily worse). A
# constant model that always predicts the expected value of $\boldsymbol{y}$,
# disregarding the input features, would get a $R^2$ score of $0.0$.
# 
# If $\tilde{\boldsymbol{y}}_i$ is the predicted value of the $i-th$ sample and $y_i$ is the corresponding true value, then the score $R^2$ is defined as

# $$
# R^2(\boldsymbol{y}, \tilde{\boldsymbol{y}}) = 1 - \frac{\sum_{i=0}^{n - 1} (y_i - \tilde{y}_i)^2}{\sum_{i=0}^{n - 1} (y_i - \bar{y})^2},
# $$

# where we have defined the mean value  of $\boldsymbol{y}$ as

# $$
# \bar{y} =  \frac{1}{n} \sum_{i=0}^{n - 1} y_i.
# $$

# Another quantity taht we will meet again in our discussions of regression analysis is 
#  the mean absolute error (MAE), a risk metric corresponding to the expected value of the absolute error loss or what we call the $l1$-norm loss. In our discussion above we presented the relative error.
# The MAE is defined as follows

# $$
# \text{MAE}(\boldsymbol{y}, \boldsymbol{\tilde{y}}) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i - \tilde{y}_i \right|.
# $$

# We present the 
# squared logarithmic (quadratic) error

# $$
# \text{MSLE}(\boldsymbol{y}, \boldsymbol{\tilde{y}}) = \frac{1}{n} \sum_{i=0}^{n - 1} (\log_e (1 + y_i) - \log_e (1 + \tilde{y}_i) )^2,
# $$

# where $\log_e (x)$ stands for the natural logarithm of $x$. This error
# estimate is best to use when targets having exponential growth, such
# as population counts, average sales of a commodity over a span of
# years etc. 
# 
# Finally, another cost function is the Huber cost function used in robust regression.
# 
# The rationale behind this possible cost function is its reduced
# sensitivity to outliers in the data set. In our discussions on
# dimensionality reduction and normalization of data we will meet other
# ways of dealing with outliers.
# 
# The Huber cost function is defined as

# $$
# H_{\delta}(\boldsymbol{a})=\left\{\begin{array}{cc}\frac{1}{2} \boldsymbol{a}^{2}& \text{for }|\boldsymbol{a}|\leq \delta\\ \delta (|\boldsymbol{a}|-\frac{1}{2}\delta ),&\text{otherwise}.\end{array}\right.
# $$

# Here $\boldsymbol{a}=\boldsymbol{y} - \boldsymbol{\tilde{y}}$.
# 
# We will discuss in more
# detail these and other functions in the various lectures.  We conclude this part with another example. Instead of 
# a linear $x$-dependence we study now a cubic polynomial and use the polynomial regression analysis tools of scikit-learn.

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x=np.linspace(0.02,0.98,200)
noise = np.asarray(random.sample((range(200)),200))
y=x**3*noise
yn=x**3*100
poly3 = PolynomialFeatures(degree=3)
X = poly3.fit_transform(x[:,np.newaxis])
clf3 = LinearRegression()
clf3.fit(X,y)

Xplot=poly3.fit_transform(x[:,np.newaxis])
poly3_plot=plt.plot(x, clf3.predict(Xplot), label='Cubic Fit')
plt.plot(x,yn, color='red', label="True Cubic")
plt.scatter(x, y, label='Data', color='orange', s=15)
plt.legend()
plt.show()

def error(a):
    for i in y:
        err=(y-yn)/yn
    return abs(np.sum(err))/len(err)

print (error(y))


# Let us now dive into  nuclear physics and remind ourselves briefly about some basic features about binding
# energies.  A basic quantity which can be measured for the ground
# states of nuclei is the atomic mass $M(N, Z)$ of the neutral atom with
# atomic mass number $A$ and charge $Z$. The number of neutrons is $N$. There are indeed several sophisticated experiments worldwide which allow us to measure this quantity to high precision (parts per million even). 
# 
# Atomic masses are usually tabulated in terms of the mass excess defined by

# $$
# \Delta M(N, Z) =  M(N, Z) - uA,
# $$

# where $u$ is the Atomic Mass Unit

# $$
# u = M(^{12}\mathrm{C})/12 = 931.4940954(57) \hspace{0.1cm} \mathrm{MeV}/c^2.
# $$

# The nucleon masses are

# $$
# m_p =  1.00727646693(9)u,
# $$

# and

# $$
# m_n = 939.56536(8)\hspace{0.1cm} \mathrm{MeV}/c^2 = 1.0086649156(6)u.
# $$

# In the [2016 mass evaluation of by W.J.Huang, G.Audi, M.Wang, F.G.Kondev, S.Naimi and X.Xu](http://nuclearmasses.org/resources_folder/Wang_2017_Chinese_Phys_C_41_030003.pdf)
# there are data on masses and decays of 3437 nuclei.
# 
# The nuclear binding energy is defined as the energy required to break
# up a given nucleus into its constituent parts of $N$ neutrons and $Z$
# protons. In terms of the atomic masses $M(N, Z)$ the binding energy is
# defined by

# $$
# BE(N, Z) = ZM_H c^2 + Nm_n c^2 - M(N, Z)c^2 ,
# $$

# where $M_H$ is the mass of the hydrogen atom and $m_n$ is the mass of the neutron.
# In terms of the mass excess the binding energy is given by

# $$
# BE(N, Z) = Z\Delta_H c^2 + N\Delta_n c^2 -\Delta(N, Z)c^2 ,
# $$

# where $\Delta_H c^2 = 7.2890$ MeV and $\Delta_n c^2 = 8.0713$ MeV.
# 
# A popular and physically intuitive model which can be used to parametrize 
# the experimental binding energies as function of $A$, is the so-called 
# **liquid drop model**. The ansatz is based on the following expression

# $$
# BE(N,Z) = a_1A-a_2A^{2/3}-a_3\frac{Z^2}{A^{1/3}}-a_4\frac{(N-Z)^2}{A},
# $$

# where $A$ stands for the number of nucleons and the $a_i$s are parameters which are determined by a fit 
# to the experimental data.  
# 
# To arrive at the above expression we have assumed that we can make the following assumptions:
# 
#  * There is a volume term $a_1A$ proportional with the number of nucleons (the energy is also an extensive quantity). When an assembly of nucleons of the same size is packed together into the smallest volume, each interior nucleon has a certain number of other nucleons in contact with it. This contribution is proportional to the volume.
# 
#  * There is a surface energy term $a_2A^{2/3}$. The assumption here is that a nucleon at the surface of a nucleus interacts with fewer other nucleons than one in the interior of the nucleus and hence its binding energy is less. This surface energy term takes that into account and is therefore negative and is proportional to the surface area.
# 
#  * There is a Coulomb energy term $a_3\frac{Z^2}{A^{1/3}}$. The electric repulsion between each pair of protons in a nucleus yields less binding. 
# 
#  * There is an asymmetry term $a_4\frac{(N-Z)^2}{A}$. This term is associated with the Pauli exclusion principle and reflects the fact that the proton-neutron interaction is more attractive on the average than the neutron-neutron and proton-proton interactions.
# 
# We could also add a so-called pairing term, which is a correction term that
# arises from the tendency of proton pairs and neutron pairs to
# occur. An even number of particles is more stable than an odd number.

# ### Organizing our data
# 
# Let us start with reading and organizing our data. 
# We start with the compilation of masses and binding energies from 2016.
# After having downloaded this file to our own computer, we are now ready to read the file and start structuring our data.
# 
# We start with preparing folders for storing our calculations and the data file over masses and binding energies. We import also various modules that we will find useful in order to present various Machine Learning methods. Here we focus mainly on the functionality of **scikit-learn**.

# In[5]:


# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

infile = open(data_path("MassEval2016.dat"),'r')


# Before we proceed, we define also a function for making our plots. You can obviously avoid this and simply set up various **matplotlib** commands every time you need them. You may however find it convenient to collect all such commands in one function and simply call this function.

# In[6]:


from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

def MakePlot(x,y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label = labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)


# Our next step is to read the data on experimental binding energies and
# reorganize them as functions of the mass number $A$, the number of
# protons $Z$ and neutrons $N$ using **pandas**.  Before we do this it is
# always useful (unless you have a binary file or other types of compressed
# data) to actually open the file and simply take a look at it!
# 
# In particular, the program that outputs the final nuclear masses is written in Fortran with a specific format. It means that we need to figure out the format and which columns contain the data we are interested in. Pandas comes with a function that reads formatted output. After having admired the file, we are now ready to start massaging it with **pandas**. The file begins with some basic format information.

# In[7]:


"""                                                                                                                         
This is taken from the data file of the mass 2016 evaluation.                                                               
All files are 3436 lines long with 124 character per line.                                                                  
       Headers are 39 lines long.                                                                                           
   col 1     :  Fortran character control: 1 = page feed  0 = line feed                                                     
   format    :  a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5                     
   These formats are reflected in the pandas widths variable below, see the statement                                       
   widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),                                                            
   Pandas has also a variable header, with length 39 in this case.                                                          
"""


# The data we are interested in are in columns 2, 3, 4 and 11, giving us
# the number of neutrons, protons, mass numbers and binding energies,
# respectively. We add also for the sake of completeness the element name. The data are in fixed-width formatted lines and we will
# covert them into the **pandas** DataFrame structure.

# In[8]:


# Read the experimental data with Pandas
Masses = pd.read_fwf(infile, usecols=(2,3,4,6,11),
              names=('N', 'Z', 'A', 'Element', 'Ebinding'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the Ebinding column won't be numeric. Coerce to float and drop these entries.
Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
Masses = Masses.dropna()
# Convert from keV to MeV.
Masses['Ebinding'] /= 1000

# Group the DataFrame by nucleon number, A.
Masses = Masses.groupby('A')
# Find the rows of the grouped DataFrame with the maximum binding energy.
Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])


# We have now read in the data, grouped them according to the variables we are interested in. 
# We see how easy it is to reorganize the data using **pandas**. If we
# were to do these operations in C/C++ or Fortran, we would have had to
# write various functions/subroutines which perform the above
# reorganizations for us.  Having reorganized the data, we can now start
# to make some simple fits using both the functionalities in **numpy** and
# **Scikit-Learn** afterwards. 
# 
# Now we define five variables which contain
# the number of nucleons $A$, the number of protons $Z$ and the number of neutrons $N$, the element name and finally the energies themselves.

# In[9]:


A = Masses['A']
Z = Masses['Z']
N = Masses['N']
Element = Masses['Element']
Energies = Masses['Ebinding']
print(Masses)


# The next step, and we will define this mathematically later, is to set up the so-called **design matrix**. We will throughout call this matrix $\boldsymbol{X}$.
# It has dimensionality $n\times p$, where $n$ is the number of data points and $p$ are the so-called predictors. In our case here they are given by the number of polynomials in $A$ we wish to include in the fit.

# In[10]:


# Now we set up the design matrix X
X = np.zeros((len(A),5))
X[:,0] = 1
X[:,1] = A
X[:,2] = A**(2.0/3.0)
X[:,3] = A**(-1.0/3.0)
X[:,4] = A**(-1.0)


# Note well that we have made life simple here. We perform a fit in
# terms of the number of nucleons only.  A more sophisticated fit can be
# done by including an explicit dependence on the number of protons and
# neutrons in the asymmetry and Coulomb terms. We leave this as an exercise to you the reader.
# 
# With **Scikit-Learn** we are now ready to use linear regression and fit our data.

# In[11]:


clf = skl.LinearRegression().fit(X, Energies)
fity = clf.predict(X)


# Pretty simple!  
# Now we can print measures of how our fit is doing, the coefficients from the fits and plot the final fit together with our data.

# In[12]:


# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(Energies, fity))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(Energies, fity))
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(Energies, fity))

Masses['Eapprox']  = fity
# Generate a plot comparing the experimental with the fitted values values.
fig, ax = plt.subplots()
ax.set_xlabel(r'$A = N + Z$')
ax.set_ylabel(r'$E_\mathrm{bind}\,/\mathrm{MeV}$')
ax.plot(Masses['A'], Masses['Ebinding'], alpha=0.7, lw=2,
            label='Ame2016')
ax.plot(Masses['A'], Masses['Eapprox'], alpha=0.7, lw=2, c='m',
            label='Fit')
ax.legend()
save_fig("Masses2016")
plt.show()


# As a teaser, let us now see how we can do this with decision trees using **Scikit-Learn**. Later we will switch to so-called **random forests**!

# In[13]:



#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regr_1=DecisionTreeRegressor(max_depth=5)
regr_2=DecisionTreeRegressor(max_depth=7)
regr_3=DecisionTreeRegressor(max_depth=9)
regr_1.fit(X, Energies)
regr_2.fit(X, Energies)
regr_3.fit(X, Energies)


y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)
y_3=regr_3.predict(X)
Masses['Eapprox'] = y_3
# Plot the results
plt.figure()
plt.plot(A, Energies, color="blue", label="Data", linewidth=2)
plt.plot(A, y_1, color="red", label="max_depth=5", linewidth=2)
plt.plot(A, y_2, color="green", label="max_depth=7", linewidth=2)
plt.plot(A, y_3, color="m", label="max_depth=9", linewidth=2)

plt.xlabel("$A$")
plt.ylabel("$E$[MeV]")
plt.title("Decision Tree Regression")
plt.legend()
save_fig("Masses2016Trees")
plt.show()
print(Masses)
print(np.mean( (Energies-y_1)**2))


# With a deeper and deeper tree level, we can almost reproduce every
# single data point by increasing the max depth of the tree.
# We can actually decide to make a decision tree which fits every single point.
# As we will
# see later, this has the benefit that we can really train a model which
# traverses every single data point. However, the price we pay is that
# we will easily overfit. That is, if we apply our model to unseen data,
# we will most likely fail miserably in our attempt at making
# predictions. As an exercise, try to make the tree level larger by adjusting the maximum depth variable. When printing out the predicition, you will note that the binding energy of every nucleus is accurately reproduced.
# 
# The **seaborn** package allows us to visualize data in an efficient way. Note that we use **scikit-learn**'s multi-layer perceptron (or feed forward neural network) 
# functionality.

# In[14]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns

X_train = X
Y_train = Energies
n_hidden_neurons = 100
epochs = 100
# store models for later use
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
sns.set()
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X_train, Y_train)
        DNN_scikit[i][j] = dnn
        train_accuracy[i][j] = dnn.score(X_train, Y_train)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()


# ## Linear Regression, basic elements
# 
# [Video of Lecture](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/h20/forelesningsvideoer/LectureAug27.mp4?vrtx=view-as-webpage).
# 
# Fitting a continuous function with linear parameterization in terms of the parameters  $\boldsymbol{\beta}$.
# * Method of choice for fitting a continuous function!
# 
# * Gives an excellent introduction to central Machine Learning features with **understandable pedagogical** links to other methods like **Neural Networks**, **Support Vector Machines** etc
# 
# * Analytical expression for the fitting parameters $\boldsymbol{\beta}$
# 
# * Analytical expressions for statistical propertiers like mean values, variances, confidence intervals and more
# 
# * Analytical relation with probabilistic interpretations 
# 
# * Easy to introduce basic concepts like bias-variance tradeoff, cross-validation, resampling and regularization techniques and many other ML topics
# 
# * Easy to code! And links well with classification problems and logistic regression and neural networks
# 
# * Allows for **easy** hands-on understanding of gradient descent methods
# 
# * and many more features
# 
# For more discussions of Ridge and Lasso regression, [Wessel van Wieringen's](https://arxiv.org/abs/1509.09169) article is highly recommended.
# Similarly, [Mehta et al's article](https://arxiv.org/abs/1803.08823) is also recommended.
# 
# Regression modeling deals with the description of  the sampling distribution of a given random variable $y$ and how it varies as function of another variable or a set of such variables $\boldsymbol{x} =[x_0, x_1,\dots, x_{n-1}]^T$. 
# The first variable is called the **dependent**, the **outcome** or the **response** variable while the set of variables $\boldsymbol{x}$ is called the independent variable, or the predictor variable or the explanatory variable. 
# 
# A regression model aims at finding a likelihood function $p(\boldsymbol{y}\vert \boldsymbol{x})$, that is the conditional distribution for $\boldsymbol{y}$ with a given $\boldsymbol{x}$. The estimation of  $p(\boldsymbol{y}\vert \boldsymbol{x})$ is made using a data set with 
# * $n$ cases $i = 0, 1, 2, \dots, n-1$ 
# 
# * Response (target, dependent or outcome) variable $y_i$ with $i = 0, 1, 2, \dots, n-1$ 
# 
# * $p$ so-called explanatory (independent or predictor) variables $\boldsymbol{x}_i=[x_{i0}, x_{i1}, \dots, x_{ip-1}]$ with $i = 0, 1, 2, \dots, n-1$ and explanatory variables running from $0$ to $p-1$. See below for more explicit examples.   
# 
#  The goal of the regression analysis is to extract/exploit relationship between $\boldsymbol{y}$ and $\boldsymbol{x}$ in or to infer causal dependencies, approximations to the likelihood functions, functional relationships and to make predictions, making fits and many other things.
# 
# Consider an experiment in which $p$ characteristics of $n$ samples are
# measured. The data from this experiment, for various explanatory variables $p$ are normally represented by a matrix  
# $\mathbf{X}$.
# 
# The matrix $\mathbf{X}$ is called the *design
# matrix*. Additional information of the samples is available in the
# form of $\boldsymbol{y}$ (also as above). The variable $\boldsymbol{y}$ is
# generally referred to as the *response variable*. The aim of
# regression analysis is to explain $\boldsymbol{y}$ in terms of
# $\boldsymbol{X}$ through a functional relationship like $y_i =
# f(\mathbf{X}_{i,\ast})$. When no prior knowledge on the form of
# $f(\cdot)$ is available, it is common to assume a linear relationship
# between $\boldsymbol{X}$ and $\boldsymbol{y}$. This assumption gives rise to
# the *linear regression model* where $\boldsymbol{\beta} = [\beta_0, \ldots,
# \beta_{p-1}]^{T}$ are the *regression parameters*. 
# 
# Linear regression gives us a set of analytical equations for the parameters $\beta_j$.
# 
# In order to understand the relation among the predictors $p$, the set of data $n$ and the target (outcome, output etc) $\boldsymbol{y}$,
# consider the model we discussed for describing nuclear binding energies. 
# 
# There we assumed that we could parametrize the data using a polynomial approximation based on the liquid drop model.
# Assuming

# $$
# BE(A) = a_0+a_1A+a_2A^{2/3}+a_3A^{-1/3}+a_4A^{-1},
# $$

# we have five predictors, that is the intercept, the $A$ dependent term, the $A^{2/3}$ term and the $A^{-1/3}$ and $A^{-1}$ terms.
# This gives $p=0,1,2,3,4$. Furthermore we have $n$ entries for each predictor. It means that our design matrix is a 
# $p\times n$ matrix $\boldsymbol{X}$.
# 
# Here the predictors are based on a model we have made. A popular data set which is widely encountered in ML applications is the
# so-called [credit card default data from Taiwan](https://www.sciencedirect.com/science/article/pii/S0957417407006719?via%3Dihub). The data set contains data on $n=30000$ credit card holders with predictors like gender, marital status, age, profession, education, etc. In total there are $24$ such predictors or attributes leading to a design matrix of dimensionality $24 \times 30000$. This is however a classification problem and we will come back to it when we discuss Logistic Regression. 
# 
# Before we proceed let us study a case from linear algebra where we aim at fitting a set of data $\boldsymbol{y}=[y_0,y_1,\dots,y_{n-1}]$. We could think of these data as a result of an experiment or a complicated numerical experiment. These data are functions of a series of variables $\boldsymbol{x}=[x_0,x_1,\dots,x_{n-1}]$, that is $y_i = y(x_i)$ with $i=0,1,2,\dots,n-1$. The variables $x_i$ could represent physical quantities like time, temperature, position etc. We assume that $y(x)$ is a smooth function. 
# 
# Since obtaining these data points may not be trivial, we want to use these data to fit a function which can allow us to make predictions for values of $y$ which are not in the present set. The perhaps simplest approach is to assume we can parametrize our function in terms of a polynomial of degree $n-1$ with $n$ points, that is

# $$
# y=y(x) \rightarrow y(x_i)=\tilde{y}_i+\epsilon_i=\sum_{j=0}^{n-1} \beta_j x_i^j+\epsilon_i,
# $$

# where $\epsilon_i$ is the error in our approximation. 
# 
# For every set of values $y_i,x_i$ we have thus the corresponding set of equations

# $$
# \begin{align*}
# y_0&=\beta_0+\beta_1x_0^1+\beta_2x_0^2+\dots+\beta_{n-1}x_0^{n-1}+\epsilon_0\\
# y_1&=\beta_0+\beta_1x_1^1+\beta_2x_1^2+\dots+\beta_{n-1}x_1^{n-1}+\epsilon_1\\
# y_2&=\beta_0+\beta_1x_2^1+\beta_2x_2^2+\dots+\beta_{n-1}x_2^{n-1}+\epsilon_2\\
# \dots & \dots \\
# y_{n-1}&=\beta_0+\beta_1x_{n-1}^1+\beta_2x_{n-1}^2+\dots+\beta_{n-1}x_{n-1}^{n-1}+\epsilon_{n-1}.\\
# \end{align*}
# $$

# Defining the vectors

# $$
# \boldsymbol{y} = [y_0,y_1, y_2,\dots, y_{n-1}]^T,
# $$

# and

# $$
# \boldsymbol{\beta} = [\beta_0,\beta_1, \beta_2,\dots, \beta_{n-1}]^T,
# $$

# and

# $$
# \boldsymbol{\epsilon} = [\epsilon_0,\epsilon_1, \epsilon_2,\dots, \epsilon_{n-1}]^T,
# $$

# and the design matrix

# $$
# \boldsymbol{X}=
# \begin{bmatrix} 
# 1& x_{0}^1 &x_{0}^2& \dots & \dots &x_{0}^{n-1}\\
# 1& x_{1}^1 &x_{1}^2& \dots & \dots &x_{1}^{n-1}\\
# 1& x_{2}^1 &x_{2}^2& \dots & \dots &x_{2}^{n-1}\\                      
# \dots& \dots &\dots& \dots & \dots &\dots\\
# 1& x_{n-1}^1 &x_{n-1}^2& \dots & \dots &x_{n-1}^{n-1}\\
# \end{bmatrix}
# $$

# we can rewrite our equations as

# $$
# \boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}+\boldsymbol{\epsilon}.
# $$

# The above design matrix is called a [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix).
# 
# We are obviously not limited to the above polynomial expansions.  We
# could replace the various powers of $x$ with elements of Fourier
# series or instead of $x_i^j$ we could have $\cos{(j x_i)}$ or $\sin{(j
# x_i)}$, or time series or other orthogonal functions.  For every set
# of values $y_i,x_i$ we can then generalize the equations to

# $$
# \begin{align*}
# y_0&=\beta_0x_{00}+\beta_1x_{01}+\beta_2x_{02}+\dots+\beta_{n-1}x_{0n-1}+\epsilon_0\\
# y_1&=\beta_0x_{10}+\beta_1x_{11}+\beta_2x_{12}+\dots+\beta_{n-1}x_{1n-1}+\epsilon_1\\
# y_2&=\beta_0x_{20}+\beta_1x_{21}+\beta_2x_{22}+\dots+\beta_{n-1}x_{2n-1}+\epsilon_2\\
# \dots & \dots \\
# y_{i}&=\beta_0x_{i0}+\beta_1x_{i1}+\beta_2x_{i2}+\dots+\beta_{n-1}x_{in-1}+\epsilon_i\\
# \dots & \dots \\
# y_{n-1}&=\beta_0x_{n-1,0}+\beta_1x_{n-1,2}+\beta_2x_{n-1,2}+\dots+\beta_{n-1}x_{n-1,n-1}+\epsilon_{n-1}.\\
# \end{align*}
# $$

# **Note that we have $p=n$ here. The matrix is symmetric. This is generally not the case!**
# 
# We redefine in turn the matrix $\boldsymbol{X}$ as

# $$
# \boldsymbol{X}=
# \begin{bmatrix} 
# x_{00}& x_{01} &x_{02}& \dots & \dots &x_{0,n-1}\\
# x_{10}& x_{11} &x_{12}& \dots & \dots &x_{1,n-1}\\
# x_{20}& x_{21} &x_{22}& \dots & \dots &x_{2,n-1}\\                      
# \dots& \dots &\dots& \dots & \dots &\dots\\
# x_{n-1,0}& x_{n-1,1} &x_{n-1,2}& \dots & \dots &x_{n-1,n-1}\\
# \end{bmatrix}
# $$

# and without loss of generality we rewrite again  our equations as

# $$
# \boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta}+\boldsymbol{\epsilon}.
# $$

# The left-hand side of this equation is kwown. Our error vector $\boldsymbol{\epsilon}$ and the parameter vector $\boldsymbol{\beta}$ are our unknow quantities. How can we obtain the optimal set of $\beta_i$ values? 
# 
# We have defined the matrix $\boldsymbol{X}$ via the equations

# $$
# \begin{align*}
# y_0&=\beta_0x_{00}+\beta_1x_{01}+\beta_2x_{02}+\dots+\beta_{n-1}x_{0n-1}+\epsilon_0\\
# y_1&=\beta_0x_{10}+\beta_1x_{11}+\beta_2x_{12}+\dots+\beta_{n-1}x_{1n-1}+\epsilon_1\\
# y_2&=\beta_0x_{20}+\beta_1x_{21}+\beta_2x_{22}+\dots+\beta_{n-1}x_{2n-1}+\epsilon_1\\
# \dots & \dots \\
# y_{i}&=\beta_0x_{i0}+\beta_1x_{i1}+\beta_2x_{i2}+\dots+\beta_{n-1}x_{in-1}+\epsilon_1\\
# \dots & \dots \\
# y_{n-1}&=\beta_0x_{n-1,0}+\beta_1x_{n-1,2}+\beta_2x_{n-1,2}+\dots+\beta_{n-1}x_{n-1,n-1}+\epsilon_{n-1}.\\
# \end{align*}
# $$

# As we noted above, we stayed with a system with the design matrix 
#  $\boldsymbol{X}\in {\mathbb{R}}^{n\times n}$, that is we have $p=n$. For reasons to come later (algorithmic arguments) we will hereafter define 
# our matrix as $\boldsymbol{X}\in {\mathbb{R}}^{n\times p}$, with the predictors refering to the column numbers and the entries $n$ being the row elements.
# 
# In our [introductory notes](https://compphysics.github.io/MachineLearning/doc/pub/How2ReadData/html/How2ReadData.html) we looked at the so-called [liquid drop model](https://en.wikipedia.org/wiki/Semi-empirical_mass_formula). Let us remind ourselves about what we did by looking at the code.
# 
# We restate the parts of the code we are most interested in.

# In[15]:


# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

infile = open(data_path("MassEval2016.dat"),'r')


# Read the experimental data with Pandas
Masses = pd.read_fwf(infile, usecols=(2,3,4,6,11),
              names=('N', 'Z', 'A', 'Element', 'Ebinding'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the Ebinding column won't be numeric. Coerce to float and drop these entries.
Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
Masses = Masses.dropna()
# Convert from keV to MeV.
Masses['Ebinding'] /= 1000

# Group the DataFrame by nucleon number, A.
Masses = Masses.groupby('A')
# Find the rows of the grouped DataFrame with the maximum binding energy.
Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])
A = Masses['A']
Z = Masses['Z']
N = Masses['N']
Element = Masses['Element']
Energies = Masses['Ebinding']

# Now we set up the design matrix X
X = np.zeros((len(A),5))
X[:,0] = 1
X[:,1] = A
X[:,2] = A**(2.0/3.0)
X[:,3] = A**(-1.0/3.0)
X[:,4] = A**(-1.0)
# Then nice printout using pandas
DesignMatrix = pd.DataFrame(X)
DesignMatrix.index = A
DesignMatrix.columns = ['1', 'A', 'A^(2/3)', 'A^(-1/3)', '1/A']
display(DesignMatrix)


# With $\boldsymbol{\beta}\in {\mathbb{R}}^{p\times 1}$, it means that we will hereafter write our equations for the approximation as

# $$
# \boldsymbol{\tilde{y}}= \boldsymbol{X}\boldsymbol{\beta},
# $$

# throughout these lectures. 
# 
# With the above we use the design matrix to define the approximation $\boldsymbol{\tilde{y}}$ via the unknown quantity $\boldsymbol{\beta}$ as

# $$
# \boldsymbol{\tilde{y}}= \boldsymbol{X}\boldsymbol{\beta},
# $$

# and in order to find the optimal parameters $\beta_i$ instead of solving the above linear algebra problem, we define a function which gives a measure of the spread between the values $y_i$ (which represent hopefully the exact values) and the parameterized values $\tilde{y}_i$, namely

# $$
# C(\boldsymbol{\beta})=\frac{1}{n}\sum_{i=0}^{n-1}\left(y_i-\tilde{y}_i\right)^2=\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)^T\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)\right\},
# $$

# or using the matrix $\boldsymbol{X}$ and in a more compact matrix-vector notation as

# $$
# C(\boldsymbol{\beta})=\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)\right\}.
# $$

# This function is one possible way to define the so-called cost function.
# 
# It is also common to define
# the function $C$ as

# $$
# C(\boldsymbol{\beta})=\frac{1}{2n}\sum_{i=0}^{n-1}\left(y_i-\tilde{y}_i\right)^2,
# $$

# since when taking the first derivative with respect to the unknown parameters $\beta$, the factor of $2$ cancels out. 
# 
# The function

# $$
# C(\boldsymbol{\beta})=\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)\right\},
# $$

# can be linked to the variance of the quantity $y_i$ if we interpret the latter as the mean value. 
# When linking (see the discussion below) with the maximum likelihood approach below, we will indeed interpret $y_i$ as a mean value

# $$
# y_{i}=\langle y_i \rangle = \beta_0x_{i,0}+\beta_1x_{i,1}+\beta_2x_{i,2}+\dots+\beta_{n-1}x_{i,n-1}+\epsilon_i,
# $$

# where $\langle y_i \rangle$ is the mean value. Keep in mind also that
# till now we have treated $y_i$ as the exact value. Normally, the
# response (dependent or outcome) variable $y_i$ the outcome of a
# numerical experiment or another type of experiment and is thus only an
# approximation to the true value. It is then always accompanied by an
# error estimate, often limited to a statistical error estimate given by
# the standard deviation discussed earlier. In the discussion here we
# will treat $y_i$ as our exact value for the response variable.
# 
# In order to find the parameters $\beta_i$ we will then minimize the spread of $C(\boldsymbol{\beta})$, that is we are going to solve the problem

# $$
# {\displaystyle \min_{\boldsymbol{\beta}\in
# {\mathbb{R}}^{p}}}\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)\right\}.
# $$

# In practical terms it means we will require

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \beta_j} = \frac{\partial }{\partial \beta_j}\left[ \frac{1}{n}\sum_{i=0}^{n-1}\left(y_i-\beta_0x_{i,0}-\beta_1x_{i,1}-\beta_2x_{i,2}-\dots-\beta_{n-1}x_{i,n-1}\right)^2\right]=0,
# $$

# which results in

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \beta_j} = -\frac{2}{n}\left[ \sum_{i=0}^{n-1}x_{ij}\left(y_i-\beta_0x_{i,0}-\beta_1x_{i,1}-\beta_2x_{i,2}-\dots-\beta_{n-1}x_{i,n-1}\right)\right]=0,
# $$

# or in a matrix-vector form as

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0 = \boldsymbol{X}^T\left( \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right).
# $$

# We can rewrite

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0 = \boldsymbol{X}^T\left( \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right),
# $$

# as

# $$
# \boldsymbol{X}^T\boldsymbol{y} = \boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\beta},
# $$

# and if the matrix $\boldsymbol{X}^T\boldsymbol{X}$ is invertible we have the solution

# $$
# \boldsymbol{\beta} =\left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}.
# $$

# We note also that since our design matrix is defined as $\boldsymbol{X}\in
# {\mathbb{R}}^{n\times p}$, the product $\boldsymbol{X}^T\boldsymbol{X} \in
# {\mathbb{R}}^{p\times p}$.  In the above case we have that $p \ll n$,
# in our case $p=5$ meaning that we end up with inverting a small
# $5\times 5$ matrix. This is a rather common situation, in many cases we end up with low-dimensional
# matrices to invert. The methods discussed here and for many other
# supervised learning algorithms like classification with logistic
# regression or support vector machines, exhibit dimensionalities which
# allow for the usage of direct linear algebra methods such as **LU** decomposition or **Singular Value Decomposition** (SVD) for finding the inverse of the matrix
# $\boldsymbol{X}^T\boldsymbol{X}$. 
# 
# **Small question**: Do you think the example we have at hand here (the nuclear binding energies) can lead to problems in inverting the matrix  $\boldsymbol{X}^T\boldsymbol{X}$? What kind of problems can we expect? 
# 
# The following matrix and vector relation will be useful here and for the rest of the course. Vectors are always written as boldfaced lower case letters and 
# matrices as upper case boldfaced letters.

# $$
# \frac{\partial (\boldsymbol{b}^T\boldsymbol{a})}{\partial \boldsymbol{a}} = \boldsymbol{b},
# $$

# and

# $$
# \frac{\partial (\boldsymbol{a}^T\boldsymbol{A}\boldsymbol{a})}{\partial \boldsymbol{a}} = \boldsymbol{a}^T(\boldsymbol{A}+\boldsymbol{A}^T),
# $$

# and

# $$
# \frac{\partial \left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)^T\left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)}{\partial \boldsymbol{s}} = -2\left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)^T\boldsymbol{A},
# $$

# These and other relations are discussed in the exercises following this chapter (see the end of the chapter).
# The latter equation is similar to the equation for the mean-squared error function we have been discussing. 
# We can then compute the second derivative of the cost function, which in our case is the second derivative
# of the means squared error. This leads to

# $$
# \frac{\partial^2 C(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}^T\partial \boldsymbol{\beta}} =\frac{2}{n}\boldsymbol{X}^T\boldsymbol{X}.
# $$

# This quantity defines the so- called the Hessian matrix.
# 
# The Hessian matrix plays an important role and is defined for the mean squared error  as

# $$
# \boldsymbol{H}=\boldsymbol{X}^T\boldsymbol{X}.
# $$

# The Hessian matrix for ordinary least squares is also proportional to
# the covariance matrix. As we will see in the chapter on Ridge and Lasso regression, This means that we can use the Singular Value Decomposition of a matrix  to find
# the eigenvalues of the covariance matrix and the Hessian matrix in
# terms of the singular values.
# 
# The residuals $\boldsymbol{\epsilon}$ are in turn given by

# $$
# \boldsymbol{\epsilon} = \boldsymbol{y}-\boldsymbol{\tilde{y}} = \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta},
# $$

# and with

# $$
# \boldsymbol{X}^T\left( \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)= 0,
# $$

# we have

# $$
# \boldsymbol{X}^T\boldsymbol{\epsilon}=\boldsymbol{X}^T\left( \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)= 0,
# $$

# meaning that the solution for $\boldsymbol{\beta}$ is the one which minimizes the residuals.  Later we will link this with the maximum likelihood approach.
# 
# Let us now return to our nuclear binding energies and simply code the above equations. 
# 
# It is rather straightforward to implement the matrix inversion and obtain the parameters $\boldsymbol{\beta}$. After having defined the matrix $\boldsymbol{X}$ we simply need to 
# write

# In[16]:


# matrix inversion to find beta
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Energies)
# and then make the prediction
ytilde = X @ beta


# Alternatively, you can use the least squares functionality in **Numpy** as

# In[17]:


fit = np.linalg.lstsq(X, Energies, rcond =None)[0]
ytildenp = np.dot(fit,X.T)


# And finally we plot our fit with and compare with data

# In[18]:


Masses['Eapprox']  = ytilde
# Generate a plot comparing the experimental with the fitted values values.
fig, ax = plt.subplots()
ax.set_xlabel(r'$A = N + Z$')
ax.set_ylabel(r'$E_\mathrm{bind}\,/\mathrm{MeV}$')
ax.plot(Masses['A'], Masses['Ebinding'], alpha=0.7, lw=2,
            label='Ame2016')
ax.plot(Masses['A'], Masses['Eapprox'], alpha=0.7, lw=2, c='m',
            label='Fit')
ax.legend()
save_fig("Masses2016OLS")
plt.show()


# We can easily test our fit by computing the $R2$ score that we discussed in connection with the functionality of **Scikit-Learn** in the introductory slides.
# Since we are not using **Scikit-Learn** here we can define our own $R2$ function as

# In[19]:


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


# and we would be using it as

# In[20]:


print(R2(Energies,ytilde))


# We can easily add our **MSE** score as

# In[21]:


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

print(MSE(Energies,ytilde))


# and finally the relative error as

# In[22]:


def RelativeError(y_data,y_model):
    return abs((y_data-y_model)/y_data)
print(RelativeError(Energies, ytilde))


# ### The $\chi^2$ function
# 
# Normally, the response (dependent or outcome) variable $y_i$ is the
# outcome of a numerical experiment or another type of experiment and is
# thus only an approximation to the true value. It is then always
# accompanied by an error estimate, often limited to a statistical error
# estimate given by the standard deviation discussed earlier. In the
# discussion here we will treat $y_i$ as our exact value for the
# response variable.
# 
# Introducing the standard deviation $\sigma_i$ for each measurement
# $y_i$, we define now the $\chi^2$ function (omitting the $1/n$ term)
# as

# $$
# \chi^2(\boldsymbol{\beta})=\frac{1}{n}\sum_{i=0}^{n-1}\frac{\left(y_i-\tilde{y}_i\right)^2}{\sigma_i^2}=\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)^T\frac{1}{\boldsymbol{\Sigma^2}}\left(\boldsymbol{y}-\boldsymbol{\tilde{y}}\right)\right\},
# $$

# where the matrix $\boldsymbol{\Sigma}$ is a diagonal matrix with $\sigma_i$ as matrix elements. 
# 
# In order to find the parameters $\beta_i$ we will then minimize the spread of $\chi^2(\boldsymbol{\beta})$ by requiring

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \beta_j} = \frac{\partial }{\partial \beta_j}\left[ \frac{1}{n}\sum_{i=0}^{n-1}\left(\frac{y_i-\beta_0x_{i,0}-\beta_1x_{i,1}-\beta_2x_{i,2}-\dots-\beta_{n-1}x_{i,n-1}}{\sigma_i}\right)^2\right]=0,
# $$

# which results in

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \beta_j} = -\frac{2}{n}\left[ \sum_{i=0}^{n-1}\frac{x_{ij}}{\sigma_i}\left(\frac{y_i-\beta_0x_{i,0}-\beta_1x_{i,1}-\beta_2x_{i,2}-\dots-\beta_{n-1}x_{i,n-1}}{\sigma_i}\right)\right]=0,
# $$

# or in a matrix-vector form as

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0 = \boldsymbol{A}^T\left( \boldsymbol{b}-\boldsymbol{A}\boldsymbol{\beta}\right).
# $$

# where we have defined the matrix $\boldsymbol{A} =\boldsymbol{X}/\boldsymbol{\Sigma}$ with matrix elements $a_{ij} = x_{ij}/\sigma_i$ and the vector $\boldsymbol{b}$ with elements $b_i = y_i/\sigma_i$.   
# 
# We can rewrite

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0 = \boldsymbol{A}^T\left( \boldsymbol{b}-\boldsymbol{A}\boldsymbol{\beta}\right),
# $$

# as

# $$
# \boldsymbol{A}^T\boldsymbol{b} = \boldsymbol{A}^T\boldsymbol{A}\boldsymbol{\beta},
# $$

# and if the matrix $\boldsymbol{A}^T\boldsymbol{A}$ is invertible we have the solution

# $$
# \boldsymbol{\beta} =\left(\boldsymbol{A}^T\boldsymbol{A}\right)^{-1}\boldsymbol{A}^T\boldsymbol{b}.
# $$

# If we then introduce the matrix

# $$
# \boldsymbol{H} =  \left(\boldsymbol{A}^T\boldsymbol{A}\right)^{-1},
# $$

# we have then the following expression for the parameters $\beta_j$ (the matrix elements of $\boldsymbol{H}$ are $h_{ij}$)

# $$
# \beta_j = \sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}\frac{y_i}{\sigma_i}\frac{x_{ik}}{\sigma_i} = \sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}b_ia_{ik}
# $$

# We state without proof the expression for the uncertainty  in the parameters $\beta_j$ as (we leave this as an exercise)

# $$
# \sigma^2(\beta_j) = \sum_{i=0}^{n-1}\sigma_i^2\left( \frac{\partial \beta_j}{\partial y_i}\right)^2,
# $$

# resulting in

# $$
# \sigma^2(\beta_j) = \left(\sum_{k=0}^{p-1}h_{jk}\sum_{i=0}^{n-1}a_{ik}\right)\left(\sum_{l=0}^{p-1}h_{jl}\sum_{m=0}^{n-1}a_{ml}\right) = h_{jj}!
# $$

# The first step here is to approximate the function $y$ with a first-order polynomial, that is we write

# $$
# y=y(x) \rightarrow y(x_i) \approx \beta_0+\beta_1 x_i.
# $$

# By computing the derivatives of $\chi^2$ with respect to $\beta_0$ and $\beta_1$ show that these are given by

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \beta_0} = -2\left[ \frac{1}{n}\sum_{i=0}^{n-1}\left(\frac{y_i-\beta_0-\beta_1x_{i}}{\sigma_i^2}\right)\right]=0,
# $$

# and

# $$
# \frac{\partial \chi^2(\boldsymbol{\beta})}{\partial \beta_1} = -\frac{2}{n}\left[ \sum_{i=0}^{n-1}x_i\left(\frac{y_i-\beta_0-\beta_1x_{i}}{\sigma_i^2}\right)\right]=0.
# $$

# For a linear fit (a first-order polynomial) we don't need to invert a matrix!!  
# Defining

# $$
# \gamma =  \sum_{i=0}^{n-1}\frac{1}{\sigma_i^2},
# $$

# $$
# \gamma_x =  \sum_{i=0}^{n-1}\frac{x_{i}}{\sigma_i^2},
# $$

# $$
# \gamma_y = \sum_{i=0}^{n-1}\left(\frac{y_i}{\sigma_i^2}\right),
# $$

# $$
# \gamma_{xx} =  \sum_{i=0}^{n-1}\frac{x_ix_{i}}{\sigma_i^2},
# $$

# $$
# \gamma_{xy} = \sum_{i=0}^{n-1}\frac{y_ix_{i}}{\sigma_i^2},
# $$

# we obtain

# $$
# \beta_0 = \frac{\gamma_{xx}\gamma_y-\gamma_x\gamma_y}{\gamma\gamma_{xx}-\gamma_x^2},
# $$

# $$
# \beta_1 = \frac{\gamma_{xy}\gamma-\gamma_x\gamma_y}{\gamma\gamma_{xx}-\gamma_x^2}.
# $$

# This approach (different linear and non-linear regression) suffers
# often from both being underdetermined and overdetermined in the
# unknown coefficients $\beta_i$.  A better approach is to use the
# Singular Value Decomposition (SVD) method discussed below. Or using
# Lasso and Ridge regression. See below.

# ### Fitting an Equation of State for Dense Nuclear Matter
# 
# Before we continue, let us introduce yet another example. We are going to fit the
# nuclear equation of state using results from many-body calculations.
# The equation of state we have made available here, as function of
# density, has been derived using modern nucleon-nucleon potentials with
# [the addition of three-body
# forces](https://www.sciencedirect.com/science/article/pii/S0370157399001106). This
# time the file is presented as a standard **csv** file.
# 
# The beginning of the Python code here is similar to what you have seen
# before, with the same initializations and declarations. We use also
# **pandas** again, rather extensively in order to organize our data.
# 
# The difference now is that we use **Scikit-Learn's** regression tools
# instead of our own matrix inversion implementation.

# In[23]:


# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

infile = open(data_path("EoS.csv"),'r')

# Read the EoS data as  csv file and organize the data into two arrays with density and energies
EoS = pd.read_csv(infile, names=('Density', 'Energy'))
EoS['Energy'] = pd.to_numeric(EoS['Energy'], errors='coerce')
EoS = EoS.dropna()
Energies = EoS['Energy']
Density = EoS['Density']
#  The design matrix now as function of various polytrops
X = np.zeros((len(Density),4))
X[:,3] = Density**(4.0/3.0)
X[:,2] = Density
X[:,1] = Density**(2.0/3.0)
X[:,0] = 1

# We use now Scikit-Learn's linear regressor and ridge regressor
# OLS part
clf = skl.LinearRegression().fit(X, Energies)
ytilde = clf.predict(X)
EoS['Eols']  = ytilde
# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(Energies, ytilde))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(Energies, ytilde))
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(Energies, ytilde))
print(clf.coef_, clf.intercept_)


fig, ax = plt.subplots()
ax.set_xlabel(r'$\rho[\mathrm{fm}^{-3}]$')
ax.set_ylabel(r'Energy per particle')
ax.plot(EoS['Density'], EoS['Energy'], alpha=0.7, lw=2,
            label='Theoretical data')
ax.plot(EoS['Density'], EoS['Eols'], alpha=0.7, lw=2, c='m',
            label='OLS')
ax.legend()
save_fig("EoSfitting")
plt.show()


# The above simple polynomial in density $\rho$ gives an excellent fit
# to the data.

# ## Splitting our Data in Training and Test data
# 
# It is normal in essentially all Machine Learning studies to split the
# data in a training set and a test set (sometimes also an additional
# validation set).  **Scikit-Learn** has an own function for this. There
# is no explicit recipe for how much data should be included as training
# data and say test data.  An accepted rule of thumb is to use
# approximately $2/3$ to $4/5$ of the data as training data. We will
# postpone a discussion of this splitting to the end of these notes and
# our discussion of the so-called **bias-variance** tradeoff. Here we
# limit ourselves to repeat the above equation of state fitting example
# but now splitting the data into a training set and a test set.
# 
# Let us study some examples. The first code here takes a simple
# one-dimensional second-order polynomial and we fit it to a
# second-order polynomial. Depending on the strength of the added noise,
# the various measures like the $R2$ score or the mean-squared error,
# the fit becomes better or worse.

# In[24]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)


#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),3))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)
# and then make the prediction
ytilde = X_train @ beta
print("Training R2")
print(R2(y_train,ytilde))
print("Training MSE")
print(MSE(y_train,ytilde))
ypredict = X_test @ beta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))


# Alternatively, you could write your own test-train splitting function as shown here.

# In[25]:


# equivalently in numpy
def train_test_split_numpy(inputs, labels, train_size, test_size):
    n_inputs = len(inputs)
    inputs_shuffled = inputs.copy()
    labels_shuffled = labels.copy()

    np.random.shuffle(inputs_shuffled)
    np.random.shuffle(labels_shuffled)

    train_end = int(n_inputs*train_size)
    X_train, X_test = inputs_shuffled[:train_end], inputs_shuffled[train_end:]
    Y_train, Y_test = labels_shuffled[:train_end], labels_shuffled[train_end:]

    return X_train, X_test, Y_train, Y_test


# But since **scikit-learn** has its own function for doing this and since
# it interfaces easily with **tensorflow** and other libraries, we
# normally recommend using the latter functionality.
# 
# As another example, we apply the training and testing split to 
# to the above equation of state fitting example
# but now splitting the data into a training set and a test set.

# In[26]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

infile = open(data_path("EoS.csv"),'r')

# Read the EoS data as  csv file and organized into two arrays with density and energies
EoS = pd.read_csv(infile, names=('Density', 'Energy'))
EoS['Energy'] = pd.to_numeric(EoS['Energy'], errors='coerce')
EoS = EoS.dropna()
Energies = EoS['Energy']
Density = EoS['Density']
#  The design matrix now as function of various polytrops
X = np.zeros((len(Density),5))
X[:,0] = 1
X[:,1] = Density**(2.0/3.0)
X[:,2] = Density
X[:,3] = Density**(4.0/3.0)
X[:,4] = Density**(5.0/3.0)
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, Energies, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
# and then make the prediction
ytilde = X_train @ beta
print("Training R2")
print(R2(y_train,ytilde))
print("Training MSE")
print(MSE(y_train,ytilde))
ypredict = X_test @ beta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))


# ## The Boston housing data example
# 
# The Boston housing  
# data set was originally a part of UCI Machine Learning Repository
# and has been removed now. The data set is now included in **Scikit-Learn**'s 
# library.  There are 506 samples and 13 feature (predictor) variables
# in this data set. The objective is to predict the value of prices of
# the house using the features (predictors) listed here.
# 
# The features/predictors are
# 1. CRIM: Per capita crime rate by town
# 
# 2. ZN: Proportion of residential land zoned for lots over 25000 square feet
# 
# 3. INDUS: Proportion of non-retail business acres per town
# 
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# 5. NOX: Nitric oxide concentration (parts per 10 million)
# 
# 6. RM: Average number of rooms per dwelling
# 
# 7. AGE: Proportion of owner-occupied units built prior to 1940
# 
# 8. DIS: Weighted distances to five Boston employment centers
# 
# 9. RAD: Index of accessibility to radial highways
# 
# 10. TAX: Full-value property tax rate per USD10000
# 
# 11. B: $1000(Bk - 0.63)^2$, where $Bk$ is the proportion of [people of African American descent] by town
# 
# 12. LSTAT: Percentage of lower status of the population
# 
# 13. MEDV: Median value of owner-occupied homes in USD 1000s

# ## Housing data, the code
# We start by importing the libraries

# In[27]:


import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns


# and load the Boston Housing DataSet from **Scikit-Learn**

# In[28]:


from sklearn.datasets import load_boston

boston_dataset = load_boston()

# boston_dataset is a dictionary
# let's check what it contains
boston_dataset.keys()


# Then we invoke Pandas

# In[29]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston['MEDV'] = boston_dataset.target


# and preprocess the data

# In[30]:


# check for missing values in all the columns
boston.isnull().sum()


# We can then visualize the data

# In[31]:


# set the size of the figure
sns.set(rc={'figure.figsize':(11.7,8.27)})

# plot a histogram showing the distribution of the target values
sns.distplot(boston['MEDV'], bins=30)
plt.show()


# It is now useful to look at the correlation matrix

# In[32]:


# compute the pair wise correlation for all columns  
correlation_matrix = boston.corr().round(2)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# From the above coorelation plot we can see that **MEDV** is strongly correlated to **LSTAT** and  **RM**. We see also that **RAD** and **TAX** are stronly correlated, but we don't include this in our features together to avoid multi-colinearity

# In[33]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# Now we start training our model

# In[34]:


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# We split the data into training and test sets

# In[35]:


from sklearn.model_selection import train_test_split

# splits the training and test data set in 80% : 20%
# assign random_state to any value.This ensures consistency.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# Then we use the linear regression functionality from **Scikit-Learn**

# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[37]:


# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(Y_test, y_test_predict)
plt.show()


# ## Reducing the number of degrees of freedom, overarching view
# 
# Many Machine Learning problems involve thousands or even millions of
# features for each training instance. Not only does this make training
# extremely slow, it can also make it much harder to find a good
# solution, as we will see. This problem is often referred to as the
# curse of dimensionality.  Fortunately, in real-world problems, it is
# often possible to reduce the number of features considerably, turning
# an intractable problem into a tractable one.
# 
# Later  we will discuss some of the most popular dimensionality reduction
# techniques: the principal component analysis (PCA), Kernel PCA, and
# Locally Linear Embedding (LLE).  
# 
# Principal component analysis and its various variants deal with the
# problem of fitting a low-dimensional [affine
# subspace](https://en.wikipedia.org/wiki/Affine_space) to a set of of
# data points in a high-dimensional space. With its family of methods it
# is one of the most used tools in data modeling, compression and
# visualization.
# 
# Before we proceed however, we will discuss how to preprocess our
# data. Till now and in connection with our previous examples we have
# not met so many cases where we are too sensitive to the scaling of our
# data. Normally the data may need a rescaling and/or may be sensitive
# to extreme values. Scaling the data renders our inputs much more
# suitable for the algorithms we want to employ.
# 
# For data sets gathered for real world applications, it is rather normal that
# different features have very different units and
# numerical scales. For example, a data set detailing health habits may include
# features such as **age** in the range $0-80$, and **caloric intake** of order $2000$.
# Many machine learning methods sensitive to the scales of the features and may perform poorly if they
# are very different scales. Therefore, it is typical to scale
# the features in a way to avoid such outlier values.
# 
# **Scikit-Learn** has several functions which allow us to rescale the
# data, normally resulting in much better results in terms of various
# accuracy scores.  The **StandardScaler** function in **Scikit-Learn**
# ensures that for each feature/predictor we study the mean value is
# zero and the variance is one (every column in the design/feature
# matrix).  This scaling has the drawback that it does not ensure that
# we have a particular maximum or minimum in our data set. Another
# function included in **Scikit-Learn** is the **MinMaxScaler** which
# ensures that all features are exactly between $0$ and $1$. The
# 
# The **Normalizer** scales each data
# point such that the feature vector has a euclidean length of one. In other words, it
# projects a data point on the circle (or sphere in the case of higher dimensions) with a
# radius of 1. This means every data point is scaled by a different number (by the
# inverse of it’s length).
# This normalization is often used when only the direction (or angle) of the data matters,
# not the length of the feature vector.
# 
# The **RobustScaler** works similarly to the StandardScaler in that it
# ensures statistical properties for each feature that guarantee that
# they are on the same scale. However, the RobustScaler uses the median
# and quartiles, instead of mean and variance. This makes the
# RobustScaler ignore data points that are very different from the rest
# (like measurement errors). These odd data points are also called
# outliers, and might often lead to trouble for other scaling
# techniques.
# 
# Many features are often scaled using standardization to improve
# performance. In **Scikit-Learn** this is given by the **StandardScaler**
# function as discussed above. It is easy however to write your own.
# Mathematically, this involves subtracting the mean and divide by the
# standard deviation over the data set, for each feature:

# $$
# x_j^{(i)} \rightarrow \frac{x_j^{(i)} - \overline{x}_j}{\sigma(x_j)},
# $$

# where $\overline{x}_j$ and $\sigma(x_j)$ are the mean and standard
# deviation, respectively, of the feature $x_j$.  This ensures that each
# feature has zero mean and unit standard deviation.  For data sets
# where we do not have the standard deviation or don't wish to calculate
# it, it is then common to simply set it to one.
# 
# Let us consider the following vanilla example where we use both
# **Scikit-Learn** and write our own function as well.  We produce a
# simple test design matrix with random numbers. Each column could then
# represent a specific feature whose mean value is subracted.

# In[38]:


import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
import pandas as pd
from IPython.display import display
np.random.seed(100)
# setting up a 10 x 5 matrix
rows = 10
cols = 5
X = np.random.randn(rows,cols)
XPandas = pd.DataFrame(X)
display(XPandas)
print(XPandas.mean())
print(XPandas.std())
XPandas = (XPandas -XPandas.mean())
display(XPandas)
#  This option does not include the standard deviation
scaler = StandardScaler(with_std=False)
scaler.fit(X)
Xscaled = scaler.transform(X)
display(XPandas-Xscaled)


# Small exercise: perform the standard scaling by including the standard deviation and compare with what Scikit-Learn gives.
# 
# Another commonly used scaling method is min-max scaling. This is very
# useful for when we want the features to lie in a certain interval. To
# scale the feature $x_j$ to the interval $[a, b]$, we can apply the
# transformation

# $$
# x_j^{(i)} \rightarrow (b-a)\frac{x_j^{(i)} - \min(x_j)}{\max(x_j) - \min(x_j)} - a
# $$

# where $\min(x_j)$ and $\max(x_j)$ return the minimum and maximum value of $x_j$ over the data set, respectively.

# ## Testing the Means Squared Error as function of Complexity
# 
# Before we proceed with a more detailed analysis of the so-called
# Bias-Variance tradeoff, we present here an example of the relation
# between model complexity and the mean squared error for the triaining
# data and the test data.
# 
# The results here tell us clearly that for the data not included in the
# training, there is an optimal model as function of the complexity of
# ourmodel (here in terms of the polynomial degree of the model).
# 
# The results here will vary as function of model complexity and the amount od data used for training. 
# 
# Our data is defined by $x\in [-3,3]$ with a total of for example $100$ data points.

# In[39]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


np.random.seed(2018)
n = 100
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    clf = model.fit(x_train,y_train)
    y_fit = clf.predict(x_train)
    y_pred = clf.predict(x_test) 
    polydegree[degree] = degree
    TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )

plt.plot(polydegree, TestError, label='Test Error')
plt.plot(polydegree, TrainError, label='Train Error')
plt.legend()
plt.show()


# ## Exercises

# ## Exercise 1: Setting up various Python environments
# 
# The first exercise here is of a mere technical art. We want you to have 
# * git as a version control software and to establish a user account on a provider like GitHub. Other providers like GitLab etc are equally fine. You can also use the University of Oslo [GitHub facilities](https://www.uio.no/tjenester/it/maskin/filer/versjonskontroll/github.html). 
# 
# * Install various Python packages
# 
# We will make extensive use of Python as programming language and its
# myriad of available libraries.  You will find
# IPython/Jupyter notebooks invaluable in your work.  You can run **R**
# codes in the Jupyter/IPython notebooks, with the immediate benefit of
# visualizing your data. You can also use compiled languages like C++,
# Rust, Fortran etc if you prefer. The focus in these lectures will be
# on Python.
# 
# If you have Python installed (we recommend Python3) and you feel
# pretty familiar with installing different packages, we recommend that
# you install the following Python packages via **pip** as 
# 
# 1. pip install numpy scipy matplotlib ipython scikit-learn sympy pandas pillow 
# 
# For **Tensorflow**, we recommend following the instructions in the text of 
# [Aurelien Geron, Hands‑On Machine Learning with Scikit‑Learn and TensorFlow, O'Reilly](http://shop.oreilly.com/product/0636920052289.do)
# 
# We will come back to **tensorflow** later. 
# 
# For Python3, replace **pip** with **pip3**.
# 
# For OSX users we recommend, after having installed Xcode, to
# install **brew**. Brew allows for a seamless installation of additional
# software via for example 
# 
# 1. brew install python3
# 
# For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution,
# you can use **pip** as well and simply install Python as 
# 
# 1. sudo apt-get install python3  (or python for Python2.7)
# 
# If you don't want to perform these operations separately and venture
# into the hassle of exploring how to set up dependencies and paths, we
# recommend two widely used distrubutions which set up all relevant
# dependencies for Python, namely 
# 
# * [Anaconda](https://docs.anaconda.com/), 
# 
# which is an open source
# distribution of the Python and R programming languages for large-scale
# data processing, predictive analytics, and scientific computing, that
# aims to simplify package management and deployment. Package versions
# are managed by the package management system **conda**. 
# 
# * [Enthought canopy](https://www.enthought.com/product/canopy/) 
# 
# is a Python
# distribution for scientific and analytic computing distribution and
# analysis environment, available for free and under a commercial
# license.
# 
# We recommend using **Anaconda** if you are not too familiar with setting paths in a terminal environment.

# ## Exercise 2: making your own data and exploring scikit-learn
# 
# We will generate our own dataset for a function $y(x)$ where $x \in [0,1]$ and defined by random numbers computed with the uniform distribution. The function $y$ is a quadratic polynomial in $x$ with added stochastic noise according to the normal distribution $\cal {N}(0,1)$.
# The following simple Python instructions define our $x$ and $y$ values (with 100 data points).

# In[40]:


x = np.random.rand(100,1)
y = 2.0+5*x*x+0.1*np.random.randn(100,1)


# 1. Write your own code (following the examples under the [regression notes](https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter1.html)) for computing the parametrization of the data set fitting a second-order polynomial. 
# 
# 2. Use thereafter **scikit-learn** (see again the examples in the regression slides) and compare with your own code.   When compairing with _scikit_learn_, make sure you set the option for the intercept to **FALSE**, see <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>. This feature will be explained in more detail during the lectures of week 35 and week 36. You can find more in <https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter3.html#more-on-rescaling-data>.
# 
# 3. Using scikit-learn, compute also the mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error defined as

# $$
# MSE(\boldsymbol{y},\boldsymbol{\tilde{y}}) = \frac{1}{n}
# \sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2,
# $$

# and the $R^2$ score function.
# If $\tilde{\boldsymbol{y}}_i$ is the predicted value of the $i-th$ sample and $y_i$ is the corresponding true value, then the score $R^2$ is defined as

# $$
# R^2(\boldsymbol{y}, \tilde{\boldsymbol{y}}) = 1 - \frac{\sum_{i=0}^{n - 1} (y_i - \tilde{y}_i)^2}{\sum_{i=0}^{n - 1} (y_i - \bar{y})^2},
# $$

# where we have defined the mean value  of $\boldsymbol{y}$ as

# $$
# \bar{y} =  \frac{1}{n} \sum_{i=0}^{n - 1} y_i.
# $$

# You can use the functionality included in scikit-learn. If you feel for it, you can use your own program and define functions which compute the above two functions. 
# Discuss the meaning of these results. Try also to vary the coefficient in front of the added stochastic noise term and discuss the quality of the fits.
# 
# <!-- --- begin solution of exercise --- -->
# **Solution.**
# The code here is an example of where we define our own design matrix and fit parameters $\beta$.

# In[41]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)


#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),3))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x**2
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# matrix inversion to find beta
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(beta)
# and then make the prediction
ytilde = X_train @ beta
print("Training R2")
print(R2(y_train,ytilde))
print("Training MSE")
print(MSE(y_train,ytilde))
ypredict = X_test @ beta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))


# <!-- --- end solution of exercise --- -->

# ## Exercise 3: Normalizing our data
# 
# A much used approach before starting to train the data is  to preprocess our
# data. Normally the data may need a rescaling and/or may be sensitive
# to extreme values. Scaling the data renders our inputs much more
# suitable for the algorithms we want to employ.
# 
# **Scikit-Learn** has several functions which allow us to rescale the
# data, normally resulting in much better results in terms of various
# accuracy scores.  The **StandardScaler** function in **Scikit-Learn**
# ensures that for each feature/predictor we study the mean value is
# zero and the variance is one (every column in the design/feature
# matrix).  This scaling has the drawback that it does not ensure that
# we have a particular maximum or minimum in our data set. Another
# function included in **Scikit-Learn** is the **MinMaxScaler** which
# ensures that all features are exactly between $0$ and $1$. The
# 
# The **Normalizer** scales each data
# point such that the feature vector has a euclidean length of one. In other words, it
# projects a data point on the circle (or sphere in the case of higher dimensions) with a
# radius of 1. This means every data point is scaled by a different number (by the
# inverse of it’s length).
# This normalization is often used when only the direction (or angle) of the data matters,
# not the length of the feature vector.
# 
# The **RobustScaler** works similarly to the StandardScaler in that it
# ensures statistical properties for each feature that guarantee that
# they are on the same scale. However, the RobustScaler uses the median
# and quartiles, instead of mean and variance. This makes the
# RobustScaler ignore data points that are very different from the rest
# (like measurement errors). These odd data points are also called
# outliers, and might often lead to trouble for other scaling
# techniques.
# 
# It also common to split the data in a **training** set and a **testing** set. A typical split is to use $80\%$ of the data for training and the rest
# for testing. This can be done as follows with our design matrix $\boldsymbol{X}$ and data $\boldsymbol{y}$ (remember to import **scikit-learn**)

# In[42]:


# split in training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# Then we can use the standard scaler to scale our data as

# In[43]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In this exercise we want you to to compute the MSE for the training
# data and the test data as function of the complexity of a polynomial,
# that is the degree of a given polynomial. We want you also to compute the $R2$ score as function of the complexity of the model for both training data and test data.  You should also run the calculation with and without scaling. 
# 
# One of 
# the aims is to reproduce Figure 2.11 of [Hastie et al](https://github.com/CompPhysics/MLErasmus/blob/master/doc/Textbooks/elementsstat.pdf).
# 
# Our data is defined by $x\in [-3,3]$ with a total of for example $100$ data points.

# In[44]:


np.random.seed()
n = 100
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)


# where $y$ is the function we want to fit with a given polynomial.
# 
# <!-- --- begin solution of exercise --- -->
# **Solution.**
# We present here the solution for the last exercise. All elements here can be used to solve exercises a) and b) as well.
# Note that in this example we have used the polynomial fitting functions of **scikit-learn**.

# In[45]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


np.random.seed(2018)
n = 30
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    clf = model.fit(x_train,y_train)
    y_fit = clf.predict(x_train)
    y_pred = clf.predict(x_test) 
    polydegree[degree] = degree
    TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )

plt.plot(polydegree, TestError, label='Test Error')
plt.plot(polydegree, TrainError, label='Train Error')
plt.legend()
plt.show()


# <!-- --- end solution of exercise --- -->

# **a)**
# Write a first code which sets up a design matrix $X$ defined by a fifth-order polynomial.  Scale your data and split it in training and test data.

# **b)**
# Perform an ordinary least squares and compute the means squared error and the $R2$ factor for the training data and the test data, with and without scaling.

# **c)**
# Add now a model which allows you to make polynomials up to degree $15$.  Perform a standard OLS fitting of the training data and compute the MSE and $R2$ for the training and test data and plot both test and training data MSE and $R2$ as functions of the polynomial degree. Compare what you see with Figure 2.11 of Hastie et al. Comment your results. For which polynomial degree do you find an optimal MSE (smallest value)?

# ## Exercise 4: Adding Ridge Regression
# 
# This exercise is a continuation of exercise 2. We will use the same function to
# generate our data set, still staying with a simple function $y(x)$
# which we want to fit using linear regression, but now extending the
# analysis to include the Ridge regression method.
# 
# We will thus again generate our own dataset for a function $y(x)$ where 
# $x \in [0,1]$ and defined by random numbers computed with the uniform
# distribution. The function $y$ is a quadratic polynomial in $x$ with
# added stochastic noise according to the normal distribution $\cal{N}(0,1)$.
# 
# The following simple Python instructions define our $x$ and $y$ values (with 100 data points).

# In[46]:


x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)


# Write your own code for the Ridge method (see chapter 3.4 of Hastie *et al.*, equations (3.43) and (3.44)) and compute the parametrization for different values of $\lambda$. Compare and analyze your results with those from exercise 3. Study the dependence on $\lambda$ while also varying the strength of the noise in your expression for $y(x)$. 
# 
# Repeat the above but using the functionality of
# **Scikit-Learn**. Compare your code with the results from
# **Scikit-Learn**. Remember to run with the same random numbers for
# generating $x$ and $y$.  Observe also that when you compare with **Scikit-Learn**, you need to pay attention to how the intercept is dealt with.
# 
# Finally, using **Scikit-Learn** or your own code, compute also the mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error defined as

# $$
# MSE(\hat{y},\hat{\tilde{y}}) = \frac{1}{n}
# \sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2,
# $$

# and the $R^2$ score function.
# If $\tilde{\hat{y}}_i$ is the predicted value of the $i-th$ sample and $y_i$ is the corresponding true value, then the score $R^2$ is defined as

# $$
# R^2(\hat{y}, \tilde{\hat{y}}) = 1 - \frac{\sum_{i=0}^{n - 1} (y_i - \tilde{y}_i)^2}{\sum_{i=0}^{n - 1} (y_i - \bar{y})^2},
# $$

# where we have defined the mean value  of $\hat{y}$ as

# $$
# \bar{y} =  \frac{1}{n} \sum_{i=0}^{n - 1} y_i.
# $$

# Discuss these quantities as functions of the variable $\lambda$ in Ridge regression.
# 
# <!-- --- begin solution of exercise --- -->
# **Solution.**
# The code here allows you to perform your own Ridge calculation and
# perform calculations for various values of the regularization
# parameter $\lambda$. This program can easily be extended upon.

# In[47]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

# number of features p (here degree of polynomial
p = 3
#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),p))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x*x
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# matrix inversion to find beta
OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print(OLSbeta)
# and then make the prediction
ytildeOLS = X_train @ OLSbeta
print("Training R2 for OLS")
print(R2(y_train,ytildeOLS))
print("Training MSE for OLS")
print(MSE(y_train,ytildeOLS))
ypredictOLS = X_test @ OLSbeta
print("Test R2 for OLS")
print(R2(y_test,ypredictOLS))
print("Test MSE OLS")
print(MSE(y_test,ypredictOLS))


# Repeat now for Ridge regression and various values of the regularization parameter
I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 20
OwnMSEPredict = np.zeros(nlambdas)
OwnMSETrain = np.zeros(nlambdas)
MSERidgePredict =  np.zeros(nlambdas)
lambdas = np.logspace(-4, 1, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    OwnRidgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
    # and then make the prediction
    OwnytildeRidge = X_train @ OwnRidgebeta
    OwnypredictRidge = X_test @ OwnRidgebeta
    OwnMSEPredict[i] = MSE(y_test,OwnypredictRidge)
    OwnMSETrain[i] = MSE(y_train,OwnytildeRidge)
    # Make the fit using Ridge from Sklearn
    RegRidge = linear_model.Ridge(lmb,fit_intercept=False)
    RegRidge.fit(X_train,y_train)
    # and then make the prediction
    ypredictRidge = RegRidge.predict(X_test)
    # Compute the MSE and print it
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)

# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), OwnMSETrain, label = 'MSE Ridge train, Own code')
plt.plot(np.log10(lambdas), OwnMSEPredict, 'r--', label = 'MSE Ridge Test, Own code')
plt.plot(np.log10(lambdas), MSERidgePredict, 'g--', label = 'MSE Ridge Test, Sklearn code')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# <!-- --- end solution of exercise --- -->

# ## Exercise 5: Analytical exercises
# 
# In this exercise we derive the expressions for various derivatives of
# products of vectors and matrices. Such derivatives are central to the
# optimization of various cost functions. Although we will often use
# automatic differentiation in actual calculations, to be able to have
# analytical expressions is extremely helpful in case we have simpler
# derivatives as well as when we analyze various properties (like second
# derivatives) of the chosen cost functions.  Vectors are always written
# as boldfaced lower case letters and matrices as upper case boldfaced
# letters.
# 
# Show that

# $$
# \frac{\partial (\boldsymbol{b}^T\boldsymbol{a})}{\partial \boldsymbol{a}} = \boldsymbol{b},
# $$

# and

# $$
# \frac{\partial (\boldsymbol{a}^T\boldsymbol{A}\boldsymbol{a})}{\partial \boldsymbol{a}} = \boldsymbol{a}^T(\boldsymbol{A}+\boldsymbol{A}^T),
# $$

# and

# $$
# \frac{\partial \left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)^T\left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)}{\partial \boldsymbol{s}} = -2\left(\boldsymbol{x}-\boldsymbol{A}\boldsymbol{s}\right)^T\boldsymbol{A},
# $$

# and finally find the second derivative of this function with respect to the vector $\boldsymbol{s}$.
# 
# <!-- --- begin solution of exercise --- -->
# **Solution.**
# In these exercises it is always useful to write out with summation indices the various quantities.
# As an example, consider the function

# $$
# f(\boldsymbol{x}) =\boldsymbol{A}\boldsymbol{x},
# $$

# which reads for a specific component $f_i$ (we define the matrix $\boldsymbol{A}$ to have dimension $n\times n$ and the vector $\boldsymbol{x}$ to have length $n$)

# $$
# f_i =\sum_{j=0}^{n-1}a_{ij}x_j,
# $$

# which leads to

# $$
# \frac{\partial f_i}{\partial x_j}= a_{ij},
# $$

# and written out in terms of the vector $\boldsymbol{x}$ we have

# $$
# \frac{\partial f(\boldsymbol{x})}{\partial \boldsymbol{x}}= \boldsymbol{A}.
# $$

# For the first derivative

# $$
# \frac{\partial (\boldsymbol{b}^T\boldsymbol{a})}{\partial \boldsymbol{a}} = \boldsymbol{b},
# $$

# we can write out the inner product as (assuming all elements are real)

# $$
# \boldsymbol{b}^T\boldsymbol{a}=\sum_i b_ia_i,
# $$

# taking the derivative

# $$
# \frac{\partial \left( \sum_i b_ia_i\right)}{\partial a_k}= b_k,
# $$

# leading to

# $$
# \frac{\partial \boldsymbol{b}^T\boldsymbol{a}}{\partial \boldsymbol{a}}= \begin{bmatrix} b_0 \\ b_1 \\ b_2 \\ \dots \\ \dots \\ b_{n-1}\end{bmatrix} = \boldsymbol{b}.
# $$

# For the second exercise we have

# $$
# \frac{\partial (\boldsymbol{a}^T\boldsymbol{A}\boldsymbol{a})}{\partial \boldsymbol{a}}.
# $$

# Defining a vector $\boldsymbol{f}=\boldsymbol{A}\boldsymbol{a}$ with components $f_i=\sum_ja_{ij}a_i$  we have

# $$
# \frac{\partial (\boldsymbol{a}^T\boldsymbol{f})}{\partial \boldsymbol{a}}=\boldsymbol{a}^T\boldsymbol{A}+\boldsymbol{f}^T=\boldsymbol{a}^T\left(\boldsymbol{A}+\boldsymbol{A}^T\right),
# $$

# since $f$ depends on $a$ and we have used the chain rule for derivatives on the derivative of $f$ with respect to $a$.
# 
# <!-- --- end solution of exercise --- -->
