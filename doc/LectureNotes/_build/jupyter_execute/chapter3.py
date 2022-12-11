#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html chapter3.do.txt  -->

# # Resampling Methods

# ## Introduction
# 
# Resampling methods are an indispensable tool in modern
# statistics. They involve repeatedly drawing samples from a training
# set and refitting a model of interest on each sample in order to
# obtain additional information about the fitted model. For example, in
# order to estimate the variability of a linear regression fit, we can
# repeatedly draw different samples from the training data, fit a linear
# regression to each new sample, and then examine the extent to which
# the resulting fits differ. Such an approach may allow us to obtain
# information that would not be available from fitting the model only
# once using the original training sample.
# 
# Two resampling methods are often used in Machine Learning analyses,
# 1. The **bootstrap method**
# 
# 2. and **Cross-Validation**
# 
# In addition there are several other methods such as the Jackknife and the Blocking methods. We will discuss in particular
# cross-validation and the bootstrap method. 
# 
# Resampling approaches can be computationally expensive, because they
# involve fitting the same statistical method multiple times using
# different subsets of the training data. However, due to recent
# advances in computing power, the computational requirements of
# resampling methods generally are not prohibitive. In this chapter, we
# discuss two of the most commonly used resampling methods,
# cross-validation and the bootstrap. Both methods are important tools
# in the practical application of many statistical learning
# procedures. For example, cross-validation can be used to estimate the
# test error associated with a given statistical learning method in
# order to evaluate its performance, or to select the appropriate level
# of flexibility. The process of evaluating a model’s performance is
# known as model assessment, whereas the process of selecting the proper
# level of flexibility for a model is known as model selection. The
# bootstrap is widely used.
# 
# * Our simulations can be treated as *computer experiments*. This is particularly the case for Monte Carlo methods
# 
# * The results can be analysed with the same statistical tools as we would use analysing experimental data.
# 
# * As in all experiments, we are looking for expectation values and an estimate of how accurate they are, i.e., possible sources for errors.

# ## Reminder on Statistics
# 
# * As in other experiments, many numerical  experiments have two classes of errors:
# 
#   * Statistical errors
# 
#   * Systematical errors
# 
# * Statistical errors can be estimated using standard tools from statistics
# 
# * Systematical errors are method specific and must be treated differently from case to case. 
# 
# The
# advantage of doing linear regression is that we actually end up with
# analytical expressions for several statistical quantities.  
# Standard least squares and Ridge regression  allow us to
# derive quantities like the variance and other expectation values in a
# rather straightforward way.
# 
# It is assumed that $\varepsilon_i
# \sim \mathcal{N}(0, \sigma^2)$ and the $\varepsilon_{i}$ are
# independent, i.e.:

# $$
# \begin{align*} 
# \mbox{Cov}(\varepsilon_{i_1},
# \varepsilon_{i_2}) & = \left\{ \begin{array}{lcc} \sigma^2 & \mbox{if}
# & i_1 = i_2, \\ 0 & \mbox{if} & i_1 \not= i_2.  \end{array} \right.
# \end{align*}
# $$

# The randomness of $\varepsilon_i$ implies that
# $\mathbf{y}_i$ is also a random variable. In particular,
# $\mathbf{y}_i$ is normally distributed, because $\varepsilon_i \sim
# \mathcal{N}(0, \sigma^2)$ and $\mathbf{X}_{i,\ast} \, \boldsymbol{\beta}$ is a
# non-random scalar. To specify the parameters of the distribution of
# $\mathbf{y}_i$ we need to calculate its first two moments. 
# 
# Recall that $\boldsymbol{X}$ is a matrix of dimensionality $n\times p$. The
# notation above $\mathbf{X}_{i,\ast}$ means that we are looking at the
# row number $i$ and perform a sum over all values $p$.
# 
# The assumption we have made here can be summarized as (and this is going to be useful when we discuss the bias-variance trade off)
# that there exists a function $f(\boldsymbol{x})$ and  a normal distributed error $\boldsymbol{\varepsilon}\sim \mathcal{N}(0, \sigma^2)$
# which describe our data

# $$
# \boldsymbol{y} = f(\boldsymbol{x})+\boldsymbol{\varepsilon}
# $$

# We approximate this function with our model from the solution of the linear regression equations, that is our
# function $f$ is approximated by $\boldsymbol{\tilde{y}}$ where we want to minimize $(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2$, our MSE, with

# $$
# \boldsymbol{\tilde{y}} = \boldsymbol{X}\boldsymbol{\beta}.
# $$

# We can calculate the expectation value of $\boldsymbol{y}$ for a given element $i$

# $$
# \begin{align*} 
# \mathbb{E}(y_i) & =
# \mathbb{E}(\mathbf{X}_{i, \ast} \, \boldsymbol{\beta}) + \mathbb{E}(\varepsilon_i)
# \, \, \, = \, \, \, \mathbf{X}_{i, \ast} \, \beta, 
# \end{align*}
# $$

# while
# its variance is

# $$
# \begin{align*} \mbox{Var}(y_i) & = \mathbb{E} \{ [y_i
# - \mathbb{E}(y_i)]^2 \} \, \, \, = \, \, \, \mathbb{E} ( y_i^2 ) -
# [\mathbb{E}(y_i)]^2  \\  & = \mathbb{E} [ ( \mathbf{X}_{i, \ast} \,
# \beta + \varepsilon_i )^2] - ( \mathbf{X}_{i, \ast} \, \boldsymbol{\beta})^2 \\ &
# = \mathbb{E} [ ( \mathbf{X}_{i, \ast} \, \boldsymbol{\beta})^2 + 2 \varepsilon_i
# \mathbf{X}_{i, \ast} \, \boldsymbol{\beta} + \varepsilon_i^2 ] - ( \mathbf{X}_{i,
# \ast} \, \beta)^2 \\  & = ( \mathbf{X}_{i, \ast} \, \boldsymbol{\beta})^2 + 2
# \mathbb{E}(\varepsilon_i) \mathbf{X}_{i, \ast} \, \boldsymbol{\beta} +
# \mathbb{E}(\varepsilon_i^2 ) - ( \mathbf{X}_{i, \ast} \, \boldsymbol{\beta})^2 
# \\ & = \mathbb{E}(\varepsilon_i^2 ) \, \, \, = \, \, \,
# \mbox{Var}(\varepsilon_i) \, \, \, = \, \, \, \sigma^2.  
# \end{align*}
# $$

# Hence, $y_i \sim \mathcal{N}( \mathbf{X}_{i, \ast} \, \boldsymbol{\beta}, \sigma^2)$, that is $\boldsymbol{y}$ follows a normal distribution with 
# mean value $\boldsymbol{X}\boldsymbol{\beta}$ and variance $\sigma^2$ (not be confused with the singular values of the SVD). 
# 
# With the OLS expressions for the parameters $\boldsymbol{\beta}$ we can evaluate the expectation value

# $$
# \mathbb{E}(\boldsymbol{\beta}) = \mathbb{E}[ (\mathbf{X}^{\top} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbb{E}[ \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T}\mathbf{X}\boldsymbol{\beta}=\boldsymbol{\beta}.
# $$

# This means that the estimator of the regression parameters is unbiased.
# 
# We can also calculate the variance
# 
# The variance of $\boldsymbol{\beta}$ is

# $$
# \begin{eqnarray*}
# \mbox{Var}(\boldsymbol{\beta}) & = & \mathbb{E} \{ [\boldsymbol{\beta} - \mathbb{E}(\boldsymbol{\beta})] [\boldsymbol{\beta} - \mathbb{E}(\boldsymbol{\beta})]^{T} \}
# \\
# & = & \mathbb{E} \{ [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} - \boldsymbol{\beta}] \, [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} - \boldsymbol{\beta}]^{T} \}
# \\
# % & = & \mathbb{E} \{ [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y}] \, [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y}]^{T} \} - \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}
# % \\
# % & = & \mathbb{E} \{ (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} \, \mathbf{Y}^{T} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1}  \} - \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}
# % \\
# & = & (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \, \mathbb{E} \{ \mathbf{Y} \, \mathbf{Y}^{T} \} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}
# \\
# & = & (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \, \{ \mathbf{X} \, \boldsymbol{\beta} \, \boldsymbol{\beta}^{T} \,  \mathbf{X}^{T} + \sigma^2 \} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}
# % \\
# % & = & (\mathbf{X}^T \mathbf{X})^{-1} \, \mathbf{X}^T \, \mathbf{X} \, \boldsymbol{\beta} \, \boldsymbol{\beta}^T \,  \mathbf{X}^T \, \mathbf{X} \, (\mathbf{X}^T % \mathbf{X})^{-1}
# % \\
# % & & + \, \, \sigma^2 \, (\mathbf{X}^T \mathbf{X})^{-1} \, \mathbf{X}^T  \, \mathbf{X} \, (\mathbf{X}^T \mathbf{X})^{-1} - \boldsymbol{\beta} \boldsymbol{\beta}^T
# \\
# & = & \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}  + \sigma^2 \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \boldsymbol{\beta} \, \boldsymbol{\beta}^{T}
# \, \, \, = \, \, \, \sigma^2 \, (\mathbf{X}^{T} \mathbf{X})^{-1},
# \end{eqnarray*}
# $$

# where we have used  that $\mathbb{E} (\mathbf{Y} \mathbf{Y}^{T}) =
# \mathbf{X} \, \boldsymbol{\beta} \, \boldsymbol{\beta}^{T} \, \mathbf{X}^{T} +
# \sigma^2 \, \mathbf{I}_{nn}$. From $\mbox{Var}(\boldsymbol{\beta}) = \sigma^2
# \, (\mathbf{X}^{T} \mathbf{X})^{-1}$, one obtains an estimate of the
# variance of the estimate of the $j$-th regression coefficient:
# $\boldsymbol{\sigma}^2 (\boldsymbol{\beta}_j ) = \boldsymbol{\sigma}^2 \sqrt{
# [(\mathbf{X}^{T} \mathbf{X})^{-1}]_{jj} }$. This may be used to
# construct a confidence interval for the estimates.
# 
# In a similar way, we can obtain analytical expressions for say the
# expectation values of the parameters $\boldsymbol{\beta}$ and their variance
# when we employ Ridge regression, allowing us again to define a confidence interval. 
# 
# It is rather straightforward to show that

# $$
# \mathbb{E} \big[ \boldsymbol{\beta}^{\mathrm{Ridge}} \big]=(\mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I}_{pp})^{-1} (\mathbf{X}^{\top} \mathbf{X})\boldsymbol{\beta}^{\mathrm{OLS}}.
# $$

# We see clearly that 
# $\mathbb{E} \big[ \boldsymbol{\beta}^{\mathrm{Ridge}} \big] \not= \boldsymbol{\beta}^{\mathrm{OLS}}$ for any $\lambda > 0$. We say then that the ridge estimator is biased.
# 
# We can also compute the variance as

# $$
# \mbox{Var}[\boldsymbol{\beta}^{\mathrm{Ridge}}]=\sigma^2[  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}  \mathbf{X}^{T} \mathbf{X} \{ [  \mathbf{X}^{\top} \mathbf{X} + \lambda \mathbf{I} ]^{-1}\}^{T},
# $$

# and it is easy to see that if the parameter $\lambda$ goes to infinity then the variance of Ridge parameters $\boldsymbol{\beta}$ goes to zero. 
# 
# With this, we can compute the difference

# $$
# \mbox{Var}[\boldsymbol{\beta}^{\mathrm{OLS}}]-\mbox{Var}(\boldsymbol{\beta}^{\mathrm{Ridge}})=\sigma^2 [  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}[ 2\lambda\mathbf{I} + \lambda^2 (\mathbf{X}^{T} \mathbf{X})^{-1} ] \{ [  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}\}^{T}.
# $$

# The difference is non-negative definite since each component of the
# matrix product is non-negative definite. 
# This means the variance we obtain with the standard OLS will always for $\lambda > 0$ be larger than the variance of $\boldsymbol{\beta}$ obtained with the Ridge estimator. This has interesting consequences when we discuss the so-called bias-variance trade-off below.

# ## Resampling methods
# 
# With all these analytical equations for both the OLS and Ridge
# regression, we will now outline how to assess a given model. This will
# lead us to a discussion of the so-called bias-variance tradeoff (see
# below) and so-called resampling methods.
# 
# One of the quantities we have discussed as a way to measure errors is
# the mean-squared error (MSE), mainly used for fitting of continuous
# functions. Another choice is the absolute error.
# 
# In the discussions below we will focus on the MSE and in particular since we will split the data into test and training data,
# we discuss the
# 1. prediction error or simply the **test error** $\mathrm{Err_{Test}}$, where we have a fixed training set and the test error is the MSE arising from the data reserved for testing. We discuss also the 
# 
# 2. training error $\mathrm{Err_{Train}}$, which is the average loss over the training data.
# 
# As our model becomes more and more complex, more of the training data tends to  be used. The training may thence adapt to more complicated structures in the data. This may lead to a decrease in the bias (see below for code example) and a slight increase of the variance for the test error.
# For a certain level of complexity the test error will reach minimum, before starting to increase again. The
# training error reaches a saturation.
# 
# Two famous
# resampling methods are the **independent bootstrap** and **the jackknife**. 
# 
# The jackknife is a special case of the independent bootstrap. Still, the jackknife was made
# popular prior to the independent bootstrap. And as the popularity of
# the independent bootstrap soared, new variants, such as **the dependent bootstrap**.
# 
# The Jackknife and independent bootstrap work for
# independent, identically distributed random variables.
# If these conditions are not
# satisfied, the methods will fail.  Yet, it should be said that if the data are
# independent, identically distributed, and we only want to estimate the
# variance of $\overline{X}$ (which often is the case), then there is no
# need for bootstrapping. 
# 
# The Jackknife works by making many replicas of the estimator $\widehat{\beta}$. 
# The jackknife is a resampling method where we systematically leave out one observation from the vector of observed values $\boldsymbol{x} = (x_1,x_2,\cdots,X_n)$. 
# Let $\boldsymbol{x}_i$ denote the vector

# $$
# \boldsymbol{x}_i = (x_1,x_2,\cdots,x_{i-1},x_{i+1},\cdots,x_n),
# $$

# which equals the vector $\boldsymbol{x}$ with the exception that observation
# number $i$ is left out. Using this notation, define
# $\widehat{\beta}_i$ to be the estimator
# $\widehat{\beta}$ computed using $\vec{X}_i$.

# In[1]:


from numpy import *
from numpy.random import randint, randn
from time import time

def jackknife(data, stat):
    n = len(data);t = zeros(n); inds = arange(n); t0 = time()
    ## 'jackknifing' by leaving out an observation for each i                                                                                                                      
    for i in range(n):
        t[i] = stat(delete(data,i) )

    # analysis                                                                                                                                                                     
    print("Runtime: %g sec" % (time()-t0)); print("Jackknife Statistics :")
    print("original           bias      std. error")
    print("%8g %14g %15g" % (stat(data),(n-1)*mean(t)/n, (n*var(t))**.5))

    return t


# Returns mean of data samples                                                                                                                                                     
def stat(data):
    return mean(data)


mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*random.randn(datapoints)
# jackknife returns the data sample                                                                                                                                                
t = jackknife(x, stat)


# ### Bootstrap
# 
# Bootstrapping is a nonparametric approach to statistical inference
# that substitutes computation for more traditional distributional
# assumptions and asymptotic results. Bootstrapping offers a number of
# advantages: 
# 1. The bootstrap is quite general, although there are some cases in which it fails.  
# 
# 2. Because it does not require distributional assumptions (such as normally distributed errors), the bootstrap can provide more accurate inferences when the data are not well behaved or when the sample size is small.  
# 
# 3. It is possible to apply the bootstrap to statistics with sampling distributions that are difficult to derive, even asymptotically. 
# 
# 4. It is relatively simple to apply the bootstrap to complex data-collection plans (such as stratified and clustered samples).
# 
# Since $\widehat{\beta} = \widehat{\beta}(\boldsymbol{X})$ is a function of random variables,
# $\widehat{\beta}$ itself must be a random variable. Thus it has
# a pdf, call this function $p(\boldsymbol{t})$. The aim of the bootstrap is to
# estimate $p(\boldsymbol{t})$ by the relative frequency of
# $\widehat{\beta}$. You can think of this as using a histogram
# in the place of $p(\boldsymbol{t})$. If the relative frequency closely
# resembles $p(\vec{t})$, then using numerics, it is straight forward to
# estimate all the interesting parameters of $p(\boldsymbol{t})$ using point
# estimators.  
# 
# In the case that $\widehat{\beta}$ has
# more than one component, and the components are independent, we use the
# same estimator on each component separately.  If the probability
# density function of $X_i$, $p(x)$, had been known, then it would have
# been straight forward to do this by: 
# 1. Drawing lots of numbers from $p(x)$, suppose we call one such set of numbers $(X_1^*, X_2^*, \cdots, X_n^*)$. 
# 
# 2. Then using these numbers, we could compute a replica of $\widehat{\beta}$ called $\widehat{\beta}^*$. 
# 
# By repeated use of (1) and (2), many
# estimates of $\widehat{\beta}$ could have been obtained. The
# idea is to use the relative frequency of $\widehat{\beta}^*$
# (think of a histogram) as an estimate of $p(\boldsymbol{t})$.
# 
# But
# unless there is enough information available about the process that
# generated $X_1,X_2,\cdots,X_n$, $p(x)$ is in general
# unknown. Therefore, [Efron in 1979](https://projecteuclid.org/euclid.aos/1176344552)  asked the
# question: What if we replace $p(x)$ by the relative frequency
# of the observation $X_i$; if we draw observations in accordance with
# the relative frequency of the observations, will we obtain the same
# result in some asymptotic sense? The answer is yes.
# 
# Instead of generating the histogram for the relative
# frequency of the observation $X_i$, just draw the values
# $(X_1^*,X_2^*,\cdots,X_n^*)$ with replacement from the vector
# $\boldsymbol{X}$. 
# 
# The independent bootstrap works like this: 
# 
# 1. Draw with replacement $n$ numbers for the observed variables $\boldsymbol{x} = (x_1,x_2,\cdots,x_n)$. 
# 
# 2. Define a vector $\boldsymbol{x}^*$ containing the values which were drawn from $\boldsymbol{x}$. 
# 
# 3. Using the vector $\boldsymbol{x}^*$ compute $\widehat{\beta}^*$ by evaluating $\widehat \beta$ under the observations $\boldsymbol{x}^*$. 
# 
# 4. Repeat this process $k$ times. 
# 
# When you are done, you can draw a histogram of the relative frequency
# of $\widehat \beta^*$. This is your estimate of the probability
# distribution $p(t)$. Using this probability distribution you can
# estimate any statistics thereof. In principle you never draw the
# histogram of the relative frequency of $\widehat{\beta}^*$. Instead
# you use the estimators corresponding to the statistic of interest. For
# example, if you are interested in estimating the variance of $\widehat
# \beta$, apply the estimator $\widehat \sigma^2$ to the values
# $\widehat \beta^*$.
# 
# Before we proceed however, we need to remind ourselves about a central
# theorem in statistics, namely the so-called **central limit theorem**.
# This theorem plays a central role in understanding why the Bootstrap
# (and other resampling methods) work so well on independent and
# identically distributed variables.
# 
# Suppose we have a PDF $p(x)$ from which we generate  a series $N$
# of averages $\langle x_i \rangle$. Each mean value $\langle x_i \rangle$
# is viewed as the average of a specific measurement, e.g., throwing 
# dice 100 times and then taking the average value, or producing a certain
# amount of random numbers. 
# For notational ease, we set $\langle x_i \rangle=x_i$ in the discussion
# which follows. 
# 
# If we compute the mean $z$ of $m$ such mean values $x_i$

# $$
# z=\frac{x_1+x_2+\dots+x_m}{m},
# $$

# the question we pose is which is the PDF of the new variable $z$.
# 
# The probability of obtaining an average value $z$ is the product of the 
# probabilities of obtaining arbitrary individual mean values $x_i$,
# but with the constraint that the average is $z$. We can express this through
# the following expression

# $$
# \tilde{p}(z)=\int dx_1p(x_1)\int dx_2p(x_2)\dots\int dx_mp(x_m)
#     \delta(z-\frac{x_1+x_2+\dots+x_m}{m}),
# $$

# where the $\delta$-function enbodies the constraint that the mean is $z$.
# All measurements that lead to each individual $x_i$ are expected to
# be independent, which in turn means that we can express $\tilde{p}$ as the 
# product of individual $p(x_i)$.  The independence assumption is important in the derivation of the central limit theorem.
# 
# If we use the integral expression for the $\delta$-function

# $$
# \delta(z-\frac{x_1+x_2+\dots+x_m}{m})=\frac{1}{2\pi}\int_{-\infty}^{\infty}
#    dq\exp{\left(iq(z-\frac{x_1+x_2+\dots+x_m}{m})\right)},
# $$

# and inserting $e^{i\mu q-i\mu q}$ where $\mu$ is the mean value
# we arrive at

# $$
# \tilde{p}(z)=\frac{1}{2\pi}\int_{-\infty}^{\infty}
#    dq\exp{\left(iq(z-\mu)\right)}\left[\int_{-\infty}^{\infty}
#    dxp(x)\exp{\left(iq(\mu-x)/m\right)}\right]^m,
# $$

# with the integral over $x$ resulting in

# $$
# \int_{-\infty}^{\infty}dxp(x)\exp{\left(iq(\mu-x)/m\right)}=
#   \int_{-\infty}^{\infty}dxp(x)
#    \left[1+\frac{iq(\mu-x)}{m}-\frac{q^2(\mu-x)^2}{2m^2}+\dots\right].
# $$

# The second term on the rhs disappears since this is just the mean and 
# employing the definition of $\sigma^2$ we have

# $$
# \int_{-\infty}^{\infty}dxp(x)e^{\left(iq(\mu-x)/m\right)}=
#   1-\frac{q^2\sigma^2}{2m^2}+\dots,
# $$

# resulting in

# $$
# \left[\int_{-\infty}^{\infty}dxp(x)\exp{\left(iq(\mu-x)/m\right)}\right]^m\approx
#   \left[1-\frac{q^2\sigma^2}{2m^2}+\dots \right]^m,
# $$

# and in the limit $m\rightarrow \infty$ we obtain

# $$
# \tilde{p}(z)=\frac{1}{\sqrt{2\pi}(\sigma/\sqrt{m})}
#     \exp{\left(-\frac{(z-\mu)^2}{2(\sigma/\sqrt{m})^2}\right)},
# $$

# which is the normal distribution with variance
# $\sigma^2_m=\sigma^2/m$, where $\sigma$ is the variance of the PDF $p(x)$
# and $\mu$ is also the mean of the PDF $p(x)$. 
# 
# Thus, the central limit theorem states that the PDF $\tilde{p}(z)$ of
# the average of $m$ random values corresponding to a PDF $p(x)$ 
# is a normal distribution whose mean is the 
# mean value of the PDF $p(x)$ and whose variance is the variance
# of the PDF $p(x)$ divided by $m$, the number of values used to compute $z$.
# 
# The central limit theorem leads to the well-known expression for the
# standard deviation, given by

# $$
# \sigma_m=
# \frac{\sigma}{\sqrt{m}}.
# $$

# The latter is true only if the average value is known exactly. This is obtained in the limit
# $m\rightarrow \infty$  only. Because the mean and the variance are measured quantities we obtain 
# the familiar expression in statistics

# $$
# \sigma_m\approx 
# \frac{\sigma}{\sqrt{m-1}}.
# $$

# In many cases however the above estimate for the standard deviation,
# in particular if correlations are strong, may be too simplistic. Keep
# in mind that we have assumed that the variables $x$ are independent
# and identically distributed. This is obviously not always the
# case. For example, the random numbers (or better pseudorandom numbers)
# we generate in various calculations do always exhibit some
# correlations.
# 
# The theorem is satisfied by a large class of PDFs. Note however that for a
# finite $m$, it is not always possible to find a closed form /analytic expression for
# $\tilde{p}(x)$.
# 
# The following code starts with a Gaussian distribution with mean value
# $\mu =100$ and variance $\sigma=15$. We use this to generate the data
# used in the bootstrap analysis. The bootstrap analysis returns a data
# set after a given number of bootstrap operations (as many as we have
# data points). This data set consists of estimated mean values for each
# bootstrap operation. The histogram generated by the bootstrap method
# shows that the distribution for these mean values is also a Gaussian,
# centered around the mean value $\mu=100$ but with standard deviation
# $\sigma/\sqrt{n}$, where $n$ is the number of bootstrap samples (in
# this case the same as the number of original data points). The value
# of the standard deviation is what we expect from the central limit
# theorem.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt

# Returns mean of bootstrap samples 
# Bootstrap algorithm
def bootstrap(data, datapoints):
    t = np.zeros(datapoints)
    n = len(data)
    # non-parametric bootstrap         
    for i in range(datapoints):
        t[i] = np.mean(data[np.random.randint(0,n,n)])
    # analysis    
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(t),np.std(t)))
    return t

# We set the mean value to 100 and the standard deviation to 15
mu, sigma = 100, 15
datapoints = 10000
# We generate random numbers according to the normal distribution
x = mu + sigma*np.random.randn(datapoints)
# bootstrap returns the data sample                                    
t = bootstrap(x, datapoints)


# We see that our new variance and from that the standard deviation, agrees with the central limit theorem.
# 
# We plot then the histogram together with a best fit for the data set.

# In[3]:


# the histogram of the bootstrapped data (normalized data if density = True)
n, binsboot, patches = plt.hist(t, 50, density=True, facecolor='red', alpha=0.75)
# add a 'best fit' line  
y = norm.pdf(binsboot, np.mean(t), np.std(t))
lt = plt.plot(binsboot, y, 'b', linewidth=1)
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.show()


# ## The bias-variance tradeoff
# 
# We will discuss the bias-variance tradeoff in the context of
# continuous predictions such as regression. However, many of the
# intuitions and ideas discussed here also carry over to classification
# tasks. Consider a dataset $\mathcal{L}$ consisting of the data
# $\mathbf{X}_\mathcal{L}=\{(y_j, \boldsymbol{x}_j), j=0\ldots n-1\}$. 
# 
# Let us assume that the true data is generated from a noisy model

# $$
# \boldsymbol{y}=f(\boldsymbol{x}) + \boldsymbol{\epsilon}
# $$

# where $\epsilon$ is normally distributed with mean zero and standard deviation $\sigma^2$.
# 
# In our derivation of the ordinary least squares method we defined then
# an approximation to the function $f$ in terms of the parameters
# $\boldsymbol{\beta}$ and the design matrix $\boldsymbol{X}$ which embody our model,
# that is $\boldsymbol{\tilde{y}}=\boldsymbol{X}\boldsymbol{\beta}$. 
# 
# Thereafter we found the parameters $\boldsymbol{\beta}$ by optimizing the means squared error via the so-called cost function

# $$
# C(\boldsymbol{X},\boldsymbol{\beta}) =\frac{1}{n}\sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2=\mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right].
# $$

# We can rewrite this as

# $$
# \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\frac{1}{n}\sum_i(f_i-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2+\frac{1}{n}\sum_i(\tilde{y}_i-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2+\sigma^2.
# $$

# The first term represents the square of the bias of the learning
# method, which can be thought of as the error caused by the simplifying
# assumptions built into the method. The second term represents the
# variance of the chosen model and finally the last terms is variance of
# the error $\boldsymbol{\epsilon}$.
# 
# To derive this equation, we need to recall that the variance of $\boldsymbol{y}$ and $\boldsymbol{\epsilon}$ are both equal to $\sigma^2$. The mean value of $\boldsymbol{\epsilon}$ is by definition equal to zero. Furthermore, the function $f$ is not a stochastic variable, idem for $\boldsymbol{\tilde{y}}$.
# We use a more compact notation in terms of the expectation value

# $$
# \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\mathbb{E}\left[(\boldsymbol{f}+\boldsymbol{\epsilon}-\boldsymbol{\tilde{y}})^2\right],
# $$

# and adding and subtracting $\mathbb{E}\left[\boldsymbol{\tilde{y}}\right]$ we get

# $$
# \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\mathbb{E}\left[(\boldsymbol{f}+\boldsymbol{\epsilon}-\boldsymbol{\tilde{y}}+\mathbb{E}\left[\boldsymbol{\tilde{y}}\right]-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2\right],
# $$

# which, using the abovementioned expectation values can be rewritten as

# $$
# \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\mathbb{E}\left[(\boldsymbol{y}-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2\right]+\mathrm{Var}\left[\boldsymbol{\tilde{y}}\right]+\sigma^2,
# $$

# that is the rewriting in terms of the so-called bias, the variance of the model $\boldsymbol{\tilde{y}}$ and the variance of $\boldsymbol{\epsilon}$.

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 500
n_boostraps = 100
degree = 18  # A quite high value, just to show.
noise = 0.1

# Make data set.
x = np.linspace(-1, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0, 0.1, x.shape)

# Hold out some test data that is never used in training.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Combine x transformation and model into one operation.
# Not neccesary, but convenient.
model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))

# The following (m x n_bootstraps) matrix holds the column vectors y_pred
# for each bootstrap iteration.
y_pred = np.empty((y_test.shape[0], n_boostraps))
for i in range(n_boostraps):
    x_, y_ = resample(x_train, y_train)

    # Evaluate the new model on the same test data each time.
    y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

# Note: Expectations and variances taken w.r.t. different training
# data sets, hence the axis=1. Subsequent means are taken across the test data
# set in order to obtain a total value, but before this we have error/bias/variance
# calculated per data point in the test set.
# Note 2: The use of keepdims=True is important in the calculation of bias as this 
# maintains the column vector form. Dropping this yields very unexpected results.
error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
print('Error:', error)
print('Bias^2:', bias)
print('Var:', variance)
print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

plt.plot(x[::5, :], y[::5, :], label='f(x)')
plt.scatter(x_test, y_test, label='Data points')
plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')
plt.legend()
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

np.random.seed(2018)

n = 40
n_boostraps = 100
maxdegree = 14


# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()

    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print('Polynomial degree:', degree)
    print('Error:', error[degree])
    print('Bias^2:', bias[degree])
    print('Var:', variance[degree])
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.show()


# The bias-variance tradeoff summarizes the fundamental tension in
# machine learning, particularly supervised learning, between the
# complexity of a model and the amount of training data needed to train
# it.  Since data is often limited, in practice it is often useful to
# use a less-complex model with higher bias, that is  a model whose asymptotic
# performance is worse than another model because it is easier to
# train and less sensitive to sampling noise arising from having a
# finite-sized training dataset (smaller variance). 
# 
# The above equations tell us that in
# order to minimize the expected test error, we need to select a
# statistical learning method that simultaneously achieves low variance
# and low bias. Note that variance is inherently a nonnegative quantity,
# and squared bias is also nonnegative. Hence, we see that the expected
# test MSE can never lie below $Var(\epsilon)$, the irreducible error.
# 
# What do we mean by the variance and bias of a statistical learning
# method? The variance refers to the amount by which our model would change if we
# estimated it using a different training data set. Since the training
# data are used to fit the statistical learning method, different
# training data sets  will result in a different estimate. But ideally the
# estimate for our model should not vary too much between training
# sets. However, if a method has high variance  then small changes in
# the training data can result in large changes in the model. In general, more
# flexible statistical methods have higher variance.
# 
# You may also find this recent [article](https://www.pnas.org/content/116/32/15849) of interest.

# In[6]:


"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()


# In[7]:


# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
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

Maxpolydegree = 30
X = np.zeros((len(Density),Maxpolydegree))
X[:,0] = 1.0
testerror = np.zeros(Maxpolydegree)
trainingerror = np.zeros(Maxpolydegree)
polynomial = np.zeros(Maxpolydegree)

trials = 100
for polydegree in range(1, Maxpolydegree):
    polynomial[polydegree] = polydegree
    for degree in range(polydegree):
        X[:,degree] = Density**(degree/3.0)

# loop over trials in order to estimate the expectation value of the MSE
    testerror[polydegree] = 0.0
    trainingerror[polydegree] = 0.0
    for samples in range(trials):
        x_train, x_test, y_train, y_test = train_test_split(X, Energies, test_size=0.2)
        model = LinearRegression(fit_intercept=False).fit(x_train, y_train)
        ypred = model.predict(x_train)
        ytilde = model.predict(x_test)
        testerror[polydegree] += mean_squared_error(y_test, ytilde)
        trainingerror[polydegree] += mean_squared_error(y_train, ypred) 

    testerror[polydegree] /= trials
    trainingerror[polydegree] /= trials
    print("Degree of polynomial: %3d"% polynomial[polydegree])
    print("Mean squared error on training data: %.8f" % trainingerror[polydegree])
    print("Mean squared error on test data: %.8f" % testerror[polydegree])

plt.plot(polynomial, np.log10(trainingerror), label='Training Error')
plt.plot(polynomial, np.log10(testerror), label='Test Error')
plt.xlabel('Polynomial degree')
plt.ylabel('log10[MSE]')
plt.legend()
plt.show()


# ## Cross-validation
# 
# When the repetitive splitting of the data set is done randomly,
# samples may accidently end up in a fast majority of the splits in
# either training or test set. Such samples may have an unbalanced
# influence on either model building or prediction evaluation. To avoid
# this $k$-fold cross-validation structures the data splitting. The
# samples are divided into $k$ more or less equally sized exhaustive and
# mutually exclusive subsets. In turn (at each split) one of these
# subsets plays the role of the test set while the union of the
# remaining subsets constitutes the training set. Such a splitting
# warrants a balanced representation of each sample in both training and
# test set over the splits. Still the division into the $k$ subsets
# involves a degree of randomness. This may be fully excluded when
# choosing $k=n$. This particular case is referred to as leave-one-out
# cross-validation (LOOCV). 
# 
# * Define a range of interest for the penalty parameter.
# 
# * Divide the data set into training and test set comprising samples $\{1, \ldots, n\} \setminus i$ and $\{ i \}$, respectively.
# 
# * Fit the linear regression model by means of ridge estimation  for each $\lambda$ in the grid using the training set, and the corresponding estimate of the error variance $\boldsymbol{\sigma}_{-i}^2(\lambda)$, as

# $$
# \begin{align*}
# \boldsymbol{\beta}_{-i}(\lambda) & =  ( \boldsymbol{X}_{-i, \ast}^{T}
# \boldsymbol{X}_{-i, \ast} + \lambda \boldsymbol{I}_{pp})^{-1}
# \boldsymbol{X}_{-i, \ast}^{T} \boldsymbol{y}_{-i}
# \end{align*}
# $$

# * Evaluate the prediction performance of these models on the test set by $\log\{L[y_i, \boldsymbol{X}_{i, \ast}; \boldsymbol{\beta}_{-i}(\lambda), \boldsymbol{\sigma}_{-i}^2(\lambda)]\}$. Or, by the prediction error $|y_i - \boldsymbol{X}_{i, \ast} \boldsymbol{\beta}_{-i}(\lambda)|$, the relative error, the error squared or the R2 score function.
# 
# * Repeat the first three steps  such that each sample plays the role of the test set once.
# 
# * Average the prediction performances of the test sets at each grid point of the penalty bias/parameter. It is an estimate of the prediction performance of the model corresponding to this value of the penalty parameter on novel data. It is defined as

# $$
# \begin{align*}
# \frac{1}{n} \sum_{i = 1}^n \log\{L[y_i, \mathbf{X}_{i, \ast}; \boldsymbol{\beta}_{-i}(\lambda), \boldsymbol{\sigma}_{-i}^2(\lambda)]\}.
# \end{align*}
# $$

# For the various values of $k$
# 
# 1. shuffle the dataset randomly.
# 
# 2. Split the dataset into $k$ groups.
# 
# 3. For each unique group:
# 
# a. Decide which group to use as set for test data
# 
# b. Take the remaining groups as a training data set
# 
# c. Fit a model on the training set and evaluate it on the test set
# 
# d. Retain the evaluation score and discard the model
# 
# 5. Summarize the model using the sample of model evaluation scores
# 
# The code here uses Ridge regression with cross-validation (CV)  resampling and $k$-fold CV in order to fit a specific polynomial.

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

# Generate the data.
nsamples = 100
x = np.random.randn(nsamples)
y = 3*x**2 + np.random.randn(nsamples)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))

i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(x):
        xtrain = x[train_inds]
        ytrain = y[train_inds]

        xtest = x[test_inds]
        ytest = y[test_inds]

        Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        ridge.fit(Xtrain, ytrain[:, np.newaxis])

        Xtest = poly.fit_transform(xtest[:, np.newaxis])
        ypred = ridge.predict(Xtest)

        scores_KFold[i,j] = np.sum((ypred - ytest[:, np.newaxis])**2)/np.size(ypred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    X = poly.fit_transform(x[:, np.newaxis])
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot and compare the slightly different ways to perform cross-validation

plt.figure()

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()


# More examples of the application of cross-validation follow here.

# In[9]:


# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


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

Maxpolydegree = 30
X = np.zeros((len(Density),Maxpolydegree))
X[:,0] = 1.0
estimated_mse_sklearn = np.zeros(Maxpolydegree)
polynomial = np.zeros(Maxpolydegree)
k =5
kfold = KFold(n_splits = k)

for polydegree in range(1, Maxpolydegree):
    polynomial[polydegree] = polydegree
    for degree in range(polydegree):
        X[:,degree] = Density**(degree/3.0)
        OLS = LinearRegression(fit_intercept=False)
# loop over trials in order to estimate the expectation value of the MSE
    estimated_mse_folds = cross_val_score(OLS, X, Energies, scoring='neg_mean_squared_error', cv=kfold)
#[:, np.newaxis]
    estimated_mse_sklearn[polydegree] = np.mean(-estimated_mse_folds)

plt.plot(polynomial, np.log10(estimated_mse_sklearn), label='Test Error')
plt.xlabel('Polynomial degree')
plt.ylabel('log10[MSE]')
plt.legend()
plt.show()


# Note that we have kept the intercept in the first column of design matrix $\boldsymbol{X}$. When we call the corresponding **Scikit-Learn** function we need thus to set the intercept to **False**. Libraries like **Scikit-Learn** normally scale the design matrix and do not fit intercept. See the discussions below.

# ## More on Rescaling data
# 
# We end this chapter by adding some words on scaling and how to deal with the intercept for regression cases.
# 
# When you are comparing your own code with for example **Scikit-Learn**'s
# library, there are some technicalities to keep in mind.  The examples
# here demonstrate some of these aspects with potential pitfalls.
# 
# The discussion here focuses on the role of the intercept, how we can
# set up the design matrix, what scaling we should use and other topics
# which tend  confuse us.
# 
# The intercept can be interpreted as the expected value of our
# target/output variables when all other predictors are set to zero.
# Thus, if we cannot assume that the expected outputs/targets are zero
# when all predictors are zero (the columns in the design matrix), it
# may be a bad idea to implement a model which penalizes the intercept.
# Furthermore, in for example Ridge and Lasso regression, the default solutions
# from the library **Scikit-Learn** (when not shrinking $\beta_0$) for the unknown parameters
# $\boldsymbol{\beta}$, are derived under the assumption that both $\boldsymbol{y}$ and
# $\boldsymbol{X}$ are zero centered, that is we subtract the mean values.
# 
# If our predictors represent different scales, then it is important to
# standardize the design matrix $\boldsymbol{X}$ by subtracting the mean of each
# column from the corresponding column and dividing the column with its
# standard deviation. Most machine learning libraries do this as a default. This means that if you compare your code with the results from a given library,
# the results may differ. 
# 
# The
# [Standardscaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# function in **Scikit-Learn** does this for us.  For the data sets we
# have been studying in our various examples, the data are in many cases
# already scaled and there is no need to scale them. You as a user of different machine learning algorithms, should always perform  a
# survey of your data, with a critical assessment of them in case you need to scale the data.
# 
# If you need to scale the data, not doing so will give an *unfair*
# penalization of the parameters since their magnitude depends on the
# scale of their corresponding predictor.
# 
# Suppose as an example that you 
# you have an input variable given by the heights of different persons.
# Human height might be measured in inches or meters or
# kilometers. If measured in kilometers, a standard linear regression
# model with this predictor would probably give a much bigger
# coefficient term, than if measured in millimeters.
# This can clearly lead to problems in evaluating the cost/loss functions.
# 
# Keep in mind that when you transform your data set before training a model, the same transformation needs to be done
# on your eventual new data set  before making a prediction. If we translate this into a Python code, it would could be implemented as follows

# In[10]:


"""
#Model training, we compute the mean value of y and X
y_train_mean = np.mean(y_train)
X_train_mean = np.mean(X_train,axis=0)
X_train = X_train - X_train_mean
y_train = y_train - y_train_mean

# The we fit our model with the training data
trained_model = some_model.fit(X_train,y_train)


#Model prediction, we need also to transform our data set used for the prediction.
X_test = X_test - X_train_mean #Use mean from training data
y_pred = trained_model(X_test)
y_pred = y_pred + y_train_mean
"""


# Let us try to understand what this may imply mathematically when we
# subtract the mean values, also known as *zero centering*. For
# simplicity, we will focus on  ordinary regression, as done in the above example.
# 
# The cost/loss function  for regression is

# $$
# C(\beta_0, \beta_1, ... , \beta_{p-1}) = \frac{1}{n}\sum_{i=0}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p-1} X_{ij}\beta_j\right)^2,.
# $$

# Recall also that we use the squared value. This expression can lead to an
# increased penalty for higher differences between predicted and
# output/target values.
# 
# What we have done is to single out the $\beta_0$ term in the
# definition of the mean squared error (MSE).  The design matrix $X$
# does in this case not contain any intercept column.  When we take the
# derivative with respect to $\beta_0$, we want the derivative to obey

# $$
# \frac{\partial C}{\partial \beta_j} = 0,
# $$

# for all $j$. For $\beta_0$ we have

# $$
# \frac{\partial C}{\partial \beta_0} = -\frac{2}{n}\sum_{i=0}^{n-1} \left(y_i - \beta_0 - \sum_{j=1}^{p-1} X_{ij} \beta_j\right).
# $$

# Multiplying away the constant $2/n$, we obtain

# $$
# \sum_{i=0}^{n-1} \beta_0 = \sum_{i=0}^{n-1}y_i - \sum_{i=0}^{n-1} \sum_{j=1}^{p-1} X_{ij} \beta_j.
# $$

# Let us specialize first to the case where we have only two parameters $\beta_0$ and $\beta_1$.
# Our result for $\beta_0$ simplifies then to

# $$
# n\beta_0 = \sum_{i=0}^{n-1}y_i - \sum_{i=0}^{n-1} X_{i1} \beta_1.
# $$

# We obtain then

# $$
# \beta_0 = \frac{1}{n}\sum_{i=0}^{n-1}y_i - \beta_1\frac{1}{n}\sum_{i=0}^{n-1} X_{i1}.
# $$

# If we define

# $$
# \mu_{\boldsymbol{x}_1}=\frac{1}{n}\sum_{i=0}^{n-1} X_{i1},
# $$

# and the mean value of the outputs as

# $$
# \mu_y=\frac{1}{n}\sum_{i=0}^{n-1}y_i,
# $$

# we have

# $$
# \beta_0 = \mu_y - \beta_1\mu_{\boldsymbol{x}_1}.
# $$

# In the general case with more parameters than $\beta_0$ and $\beta_1$, we have

# $$
# \beta_0 = \frac{1}{n}\sum_{i=0}^{n-1}y_i - \frac{1}{n}\sum_{i=0}^{n-1}\sum_{j=1}^{p-1} X_{ij}\beta_j.
# $$

# We can rewrite the latter equation as

# $$
# \beta_0 = \frac{1}{n}\sum_{i=0}^{n-1}y_i - \sum_{j=1}^{p-1} \mu_{\boldsymbol{x}_j}\beta_j,
# $$

# where we have defined

# $$
# \mu_{\boldsymbol{x}_j}=\frac{1}{n}\sum_{i=0}^{n-1} X_{ij},
# $$

# the mean value for all elements of the column vector $\boldsymbol{x}_j$.
# 
# Replacing $y_i$ with $y_i - y_i - \overline{\boldsymbol{y}}$ and centering also our design matrix results in a cost function (in vector-matrix disguise)

# $$
# C(\boldsymbol{\beta}) = (\boldsymbol{\tilde{y}} - \tilde{X}\boldsymbol{\beta})^T(\boldsymbol{\tilde{y}} - \tilde{X}\boldsymbol{\beta}).
# $$

# If we minimize with respect to $\boldsymbol{\beta}$ we have then

# $$
# \hat{\boldsymbol{\beta}} = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T\boldsymbol{\tilde{y}},
# $$

# where $\boldsymbol{\tilde{y}} = \boldsymbol{y} - \overline{\boldsymbol{y}}$
# and $\tilde{X}_{ij} = X_{ij} - \frac{1}{n}\sum_{k=0}^{n-1}X_{kj}$.
# 
# For Ridge regression we need to add $\lambda \boldsymbol{\beta}^T\boldsymbol{\beta}$ to the cost function and get then

# $$
# \hat{\boldsymbol{\beta}} = (\tilde{X}^T\tilde{X} + \lambda I)^{-1}\tilde{X}^T\boldsymbol{\tilde{y}}.
# $$

# What does this mean? And why do we insist on all this? Let us look at some examples.
# 
# This code shows a simple first-order fit to a data set using the above transformed data, where we consider the role of the intercept first, by either excluding it or including it (*code example thanks to  Øyvind Sigmundson Schøyen*). Here our scaling of the data is done by subtracting the mean values only.
# Note also that we do not split the data into training and test.

# In[11]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


np.random.seed(2021)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


true_beta = [2, 0.5, 3.7]

x = np.linspace(0, 1, 11)
y = np.sum(
    np.asarray([x ** p * b for p, b in enumerate(true_beta)]), axis=0
) + 0.1 * np.random.normal(size=len(x))

degree = 3
X = np.zeros((len(x), degree))

# Include the intercept in the design matrix
for p in range(degree):
    X[:, p] = x ** p

beta = fit_beta(X, y)

# Intercept is included in the design matrix
skl = LinearRegression(fit_intercept=False).fit(X, y)

print(f"True beta: {true_beta}")
print(f"Fitted beta: {beta}")
print(f"Sklearn fitted beta: {skl.coef_}")
ypredictOwn = X @ beta
ypredictSKL = skl.predict(X)
print(f"MSE with intercept column")
print(MSE(y,ypredictOwn))
print(f"MSE with intercept column from SKL")
print(MSE(y,ypredictSKL))


plt.figure()
plt.scatter(x, y, label="Data")
plt.plot(x, X @ beta, label="Fit")
plt.plot(x, skl.predict(X), label="Sklearn (fit_intercept=False)")


# Do not include the intercept in the design matrix
X = np.zeros((len(x), degree - 1))

for p in range(degree - 1):
    X[:, p] = x ** (p + 1)

# Intercept is not included in the design matrix
skl = LinearRegression(fit_intercept=True).fit(X, y)

# Use centered values for X and y when computing coefficients
y_offset = np.average(y, axis=0)
X_offset = np.average(X, axis=0)

beta = fit_beta(X - X_offset, y - y_offset)
intercept = np.mean(y_offset - X_offset @ beta)

print(f"Manual intercept: {intercept}")
print(f"Fitted beta (wiothout intercept): {beta}")
print(f"Sklearn intercept: {skl.intercept_}")
print(f"Sklearn fitted beta (without intercept): {skl.coef_}")
ypredictOwn = X @ beta
ypredictSKL = skl.predict(X)
print(f"MSE with Manual intercept")
print(MSE(y,ypredictOwn+intercept))
print(f"MSE with Sklearn intercept")
print(MSE(y,ypredictSKL))

plt.plot(x, X @ beta + intercept, "--", label="Fit (manual intercept)")
plt.plot(x, skl.predict(X), "--", label="Sklearn (fit_intercept=True)")
plt.grid()
plt.legend()

plt.show()


# The intercept is the value of our output/target variable
# when all our features are zero and our function crosses the $y$-axis (for a one-dimensional case). 
# 
# Printing the MSE, we see first that both methods give the same MSE, as
# they should.  However, when we move to for example Ridge regression,
# the way we treat the intercept may give a larger or smaller MSE,
# meaning that the MSE can be penalized by the value of the
# intercept. Not including the intercept in the fit, means that the
# regularization term does not include $\beta_0$. For different values
# of $\lambda$, this may lead to different MSE values. 
# 
# To remind the reader, the regularization term, with the intercept in Ridge regression, is given by

# $$
# \lambda \vert\vert \boldsymbol{\beta} \vert\vert_2^2 = \lambda \sum_{j=0}^{p-1}\beta_j^2,
# $$

# but when we take out the intercept, this equation becomes

# $$
# \lambda \vert\vert \boldsymbol{\beta} \vert\vert_2^2 = \lambda \sum_{j=1}^{p-1}\beta_j^2.
# $$

# For Lasso regression we have

# $$
# \lambda \vert\vert \boldsymbol{\beta} \vert\vert_1 = \lambda \sum_{j=1}^{p-1}\vert\beta_j\vert.
# $$

# It means that, when scaling the design matrix and the outputs/targets,
# by subtracting the mean values, we have an optimization problem which
# is not penalized by the intercept. The MSE value can then be smaller
# since it focuses only on the remaining quantities. If we however bring
# back the intercept, we will get a MSE which then contains the
# intercept.
# 
# Armed with this wisdom, we attempt first to simply set the intercept equal to **False** in our implementation of Ridge regression for our well-known  vanilla data set.

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(3155)

n = 100
x = np.random.rand(n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)

Maxpolydegree = 20
X = np.zeros((n,Maxpolydegree))
#We include explicitely the intercept column
for degree in range(Maxpolydegree):
    X[:,degree] = x**degree
# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

p = Maxpolydegree
I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 6
MSEOwnRidgePredict = np.zeros(nlambdas)
MSERidgePredict = np.zeros(nlambdas)
lambdas = np.logspace(-4, 2, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    OwnRidgeBeta = np.linalg.pinv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
    # Note: we include the intercept column and no scaling
    RegRidge = linear_model.Ridge(lmb,fit_intercept=False)
    RegRidge.fit(X_train,y_train)
    # and then make the prediction
    ytildeOwnRidge = X_train @ OwnRidgeBeta
    ypredictOwnRidge = X_test @ OwnRidgeBeta
    ytildeRidge = RegRidge.predict(X_train)
    ypredictRidge = RegRidge.predict(X_test)
    MSEOwnRidgePredict[i] = MSE(y_test,ypredictOwnRidge)
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)
    print("Beta values for own Ridge implementation")
    print(OwnRidgeBeta)
    print("Beta values for Scikit-Learn Ridge implementation")
    print(RegRidge.coef_)
    print("MSE values for own Ridge implementation")
    print(MSEOwnRidgePredict[i])
    print("MSE values for Scikit-Learn Ridge implementation")
    print(MSERidgePredict[i])

# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSEOwnRidgePredict, 'r', label = 'MSE own Ridge Test')
plt.plot(np.log10(lambdas), MSERidgePredict, 'g', label = 'MSE Ridge Test')

plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# The results here agree when we force **Scikit-Learn**'s Ridge function to include the first column in our design matrix.
# We see that the results agree very well. Here we have thus explicitely included the intercept column in the design matrix.
# What happens if we do not include the intercept in our fit?
# Let us see how we can change this code by zero centering.

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
np.random.seed(315)

n = 100
x = np.random.rand(n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)

Maxpolydegree = 20
X = np.zeros((n,Maxpolydegree-1))

for degree in range(1,Maxpolydegree): #No intercept column
    X[:,degree-1] = x**(degree)

# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#For our own implementation, we will need to deal with the intercept by centering the design matrix and the target variable
X_train_mean = np.mean(X_train,axis=0)
#Center by removing mean from each feature
X_train_scaled = X_train - X_train_mean 
X_test_scaled = X_test - X_train_mean
#The model intercept (called y_scaler) is given by the mean of the target variable (IF X is centered)
#Remove the intercept from the training data.
y_scaler = np.mean(y_train)           
y_train_scaled = y_train - y_scaler   

p = Maxpolydegree-1
I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 6
MSEOwnRidgePredict = np.zeros(nlambdas)
MSERidgePredict = np.zeros(nlambdas)

lambdas = np.logspace(-4, 2, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    OwnRidgeBeta = np.linalg.pinv(X_train_scaled.T @ X_train_scaled+lmb*I) @ X_train_scaled.T @ (y_train_scaled)
    intercept_ = y_scaler - X_train_mean@OwnRidgeBeta #The intercept can be shifted so the model can predict on uncentered data
    #Add intercept to prediction
    ypredictOwnRidge = X_test_scaled @ OwnRidgeBeta + y_scaler 
    RegRidge = linear_model.Ridge(lmb)
    RegRidge.fit(X_train,y_train)
    ypredictRidge = RegRidge.predict(X_test)
    MSEOwnRidgePredict[i] = MSE(y_test,ypredictOwnRidge)
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)
    print("Beta values for own Ridge implementation")
    print(OwnRidgeBeta) #Intercept is given by mean of target variable
    print("Beta values for Scikit-Learn Ridge implementation")
    print(RegRidge.coef_)
    print('Intercept from own implementation:')
    print(intercept_)
    print('Intercept from Scikit-Learn Ridge implementation')
    print(RegRidge.intercept_)
    print("MSE values for own Ridge implementation")
    print(MSEOwnRidgePredict[i])
    print("MSE values for Scikit-Learn Ridge implementation")
    print(MSERidgePredict[i])


# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSEOwnRidgePredict, 'b--', label = 'MSE own Ridge Test')
plt.plot(np.log10(lambdas), MSERidgePredict, 'g--', label = 'MSE SL Ridge Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# We see here, when compared to the code which includes explicitely the
# intercept column, that our MSE value is actually smaller. This is
# because the regularization term does not include the intercept value
# $\beta_0$ in the fitting.  This applies to Lasso regularization as
# well.  It means that our optimization is now done only with the
# centered matrix and/or vector that enter the fitting procedure. Note
# also that the problem with the intercept occurs mainly in these type
# of polynomial fitting problem.
# 
# The next example is indeed an example where all these discussions about the role of intercept are not present.

# ## More complicated Example: The Ising model
# 
# The one-dimensional Ising model with nearest neighbor interaction, no
# external field and a constant coupling constant $J$ is given by

# <!-- Equation labels as ordinary links -->
# <div id="_auto1"></div>
# 
# $$
# \begin{equation}
#     H = -J \sum_{k}^L s_k s_{k + 1},
# \label{_auto1} \tag{1}
# \end{equation}
# $$

# where $s_i \in \{-1, 1\}$ and $s_{N + 1} = s_1$. The number of spins
# in the system is determined by $L$. For the one-dimensional system
# there is no phase transition.
# 
# We will look at a system of $L = 40$ spins with a coupling constant of
# $J = 1$. To get enough training data we will generate 10000 states
# with their respective energies.

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy.linalg as scl
from sklearn.model_selection import train_test_split
import tqdm
sns.set(color_codes=True)
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

L = 40
n = int(1e4)

spins = np.random.choice([-1, 1], size=(n, L))
J = 1.0

energies = np.zeros(n)

for i in range(n):
    energies[i] = - J * np.dot(spins[i], np.roll(spins[i], 1))


# Here we use ordinary least squares
# regression to predict the energy for the nearest neighbor
# one-dimensional Ising model on a ring, i.e., the endpoints wrap
# around. We will use linear regression to fit a value for
# the coupling constant to achieve this.
# 
# A more general form for the one-dimensional Ising model is

# <!-- Equation labels as ordinary links -->
# <div id="_auto2"></div>
# 
# $$
# \begin{equation}
#     H = - \sum_j^L \sum_k^L s_j s_k J_{jk}.
# \label{_auto2} \tag{2}
# \end{equation}
# $$

# Here we allow for interactions beyond the nearest neighbors and a state dependent
# coupling constant. This latter expression can be formulated as
# a matrix-product

# <!-- Equation labels as ordinary links -->
# <div id="_auto3"></div>
# 
# $$
# \begin{equation}
#     \boldsymbol{H} = \boldsymbol{X} J,
# \label{_auto3} \tag{3}
# \end{equation}
# $$

# where $X_{jk} = s_j s_k$ and $J$ is a matrix which consists of the
# elements $-J_{jk}$. This form of writing the energy fits perfectly
# with the form utilized in linear regression, that is

# <!-- Equation labels as ordinary links -->
# <div id="_auto4"></div>
# 
# $$
# \begin{equation}
#     \boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\epsilon},
# \label{_auto4} \tag{4}
# \end{equation}
# $$

# We split the data in training and test data as discussed in the previous example

# In[15]:


X = np.zeros((n, L ** 2))
for i in range(n):
    X[i] = np.outer(spins[i], spins[i]).ravel()
y = energies
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In the ordinary least squares method we choose the cost function

# <!-- Equation labels as ordinary links -->
# <div id="_auto5"></div>
# 
# $$
# \begin{equation}
#     C(\boldsymbol{X}, \boldsymbol{\beta})= \frac{1}{n}\left\{(\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y})^T(\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y})\right\}.
# \label{_auto5} \tag{5}
# \end{equation}
# $$

# We then find the extremal point of $C$ by taking the derivative with respect to $\boldsymbol{\beta}$ as discussed above.
# This yields the expression for $\boldsymbol{\beta}$ to be

# $$
# \boldsymbol{\beta} = \frac{\boldsymbol{X}^T \boldsymbol{y}}{\boldsymbol{X}^T \boldsymbol{X}},
# $$

# which immediately imposes some requirements on $\boldsymbol{X}$ as there must exist
# an inverse of $\boldsymbol{X}^T \boldsymbol{X}$. If the expression we are modeling contains an
# intercept, i.e., a constant term, we must make sure that the
# first column of $\boldsymbol{X}$ consists of $1$. We do this here

# In[16]:


X_train_own = np.concatenate(
    (np.ones(len(X_train))[:, np.newaxis], X_train),
    axis=1
)
X_test_own = np.concatenate(
    (np.ones(len(X_test))[:, np.newaxis], X_test),
    axis=1
)


# Doing the inversion directly turns out to be a bad idea since the matrix
# $\boldsymbol{X}^T\boldsymbol{X}$ is singular. An alternative approach is to use the **singular
# value decomposition**. Using the definition of the Moore-Penrose
# pseudoinverse we can write the equation for $\boldsymbol{\beta}$ as

# $$
# \boldsymbol{\beta} = \boldsymbol{X}^{+}\boldsymbol{y},
# $$

# where the pseudoinverse of $\boldsymbol{X}$ is given by

# $$
# \boldsymbol{X}^{+} = \frac{\boldsymbol{X}^T}{\boldsymbol{X}^T\boldsymbol{X}}.
# $$

# Using singular value decomposition we can decompose the matrix  $\boldsymbol{X} = \boldsymbol{U}\boldsymbol{\Sigma} \boldsymbol{V}^T$,
# where $\boldsymbol{U}$ and $\boldsymbol{V}$ are orthogonal(unitary) matrices and $\boldsymbol{\Sigma}$ contains the singular values (more details below).
# where $X^{+} = V\Sigma^{+} U^T$. This reduces the equation for
# $\omega$ to

# <!-- Equation labels as ordinary links -->
# <div id="_auto6"></div>
# 
# $$
# \begin{equation}
#     \boldsymbol{\beta} = \boldsymbol{V}\boldsymbol{\Sigma}^{+} \boldsymbol{U}^T \boldsymbol{y}.
# \label{_auto6} \tag{6}
# \end{equation}
# $$

# Note that solving this equation by actually doing the pseudoinverse
# (which is what we will do) is not a good idea as this operation scales
# as $\mathcal{O}(n^3)$, where $n$ is the number of elements in a
# general matrix. Instead, doing $QR$-factorization and solving the
# linear system as an equation would reduce this down to
# $\mathcal{O}(n^2)$ operations.

# In[17]:


def ols_svd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, s, v = scl.svd(x)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ y


# In[18]:


beta = ols_svd(X_train_own,y_train)


# When extracting the $J$-matrix  we need to make sure that we remove the intercept, as is done here

# In[19]:


J = beta[1:].reshape(L, L)


# A way of looking at the coefficients in $J$ is to plot the matrices as images.

# In[20]:


fig = plt.figure(figsize=(20, 14))
im = plt.imshow(J, **cmap_args)
plt.title("OLS", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb = fig.colorbar(im)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
plt.show()


# It is interesting to note that OLS
# considers both $J_{j, j + 1} = -0.5$ and $J_{j, j - 1} = -0.5$ as
# valid matrix elements for $J$.
# In our discussion below on hyperparameters and Ridge and Lasso regression we will see that
# this problem can be removed, partly and only with Lasso regression. 
# 
# In this case our matrix inversion was actually possible. The obvious question now is what is the mathematics behind the SVD?
# 
# Let us now 
# focus on Ridge and Lasso regression as well. We repeat some of the
# basic parts of the Ising model and the setup of the training and test
# data.  The one-dimensional Ising model with nearest neighbor
# interaction, no external field and a constant coupling constant $J$ is
# given by

# <!-- Equation labels as ordinary links -->
# <div id="_auto7"></div>
# 
# $$
# \begin{equation}
#     H = -J \sum_{k}^L s_k s_{k + 1},
# \label{_auto7} \tag{7}
# \end{equation}
# $$

# where $s_i \in \{-1, 1\}$ and $s_{N + 1} = s_1$. The number of spins in the system is determined by $L$. For the one-dimensional system there is no phase transition.
# 
# We will look at a system of $L = 40$ spins with a coupling constant of $J = 1$. To get enough training data we will generate 10000 states with their respective energies.

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import scipy.linalg as scl
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
import tqdm
sns.set(color_codes=True)
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

L = 40
n = int(1e4)

spins = np.random.choice([-1, 1], size=(n, L))
J = 1.0

energies = np.zeros(n)

for i in range(n):
    energies[i] = - J * np.dot(spins[i], np.roll(spins[i], 1))


# A more general form for the one-dimensional Ising model is

# <!-- Equation labels as ordinary links -->
# <div id="_auto8"></div>
# 
# $$
# \begin{equation}
#     H = - \sum_j^L \sum_k^L s_j s_k J_{jk}.
# \label{_auto8} \tag{8}
# \end{equation}
# $$

# Here we allow for interactions beyond the nearest neighbors and a more
# adaptive coupling matrix. This latter expression can be formulated as
# a matrix-product on the form

# <!-- Equation labels as ordinary links -->
# <div id="_auto9"></div>
# 
# $$
# \begin{equation}
#     H = X J,
# \label{_auto9} \tag{9}
# \end{equation}
# $$

# where $X_{jk} = s_j s_k$ and $J$ is the matrix consisting of the
# elements $-J_{jk}$. This form of writing the energy fits perfectly
# with the form utilized in linear regression, viz.

# <!-- Equation labels as ordinary links -->
# <div id="_auto10"></div>
# 
# $$
# \begin{equation}
#     \boldsymbol{y} = \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}.
# \label{_auto10} \tag{10}
# \end{equation}
# $$

# We organize the data as we did above

# In[22]:


X = np.zeros((n, L ** 2))
for i in range(n):
    X[i] = np.outer(spins[i], spins[i]).ravel()
y = energies
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.96)

X_train_own = np.concatenate(
    (np.ones(len(X_train))[:, np.newaxis], X_train),
    axis=1
)

X_test_own = np.concatenate(
    (np.ones(len(X_test))[:, np.newaxis], X_test),
    axis=1
)


# We will do all fitting with **Scikit-Learn**,

# In[23]:


clf = skl.LinearRegression().fit(X_train, y_train)


# When  extracting the $J$-matrix we make sure to remove the intercept

# In[24]:


J_sk = clf.coef_.reshape(L, L)


# And then we plot the results

# In[25]:


fig = plt.figure(figsize=(20, 14))
im = plt.imshow(J_sk, **cmap_args)
plt.title("LinearRegression from Scikit-learn", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb = fig.colorbar(im)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)
plt.show()


# The results agree perfectly with our previous discussion where we used our own code.
# 
# Having explored the ordinary least squares we move on to ridge
# regression. In ridge regression we include a **regularizer**. This
# involves a new cost function which leads to a new estimate for the
# weights $\boldsymbol{\beta}$. This results in a penalized regression problem. The
# cost function is given by

# <!-- Equation labels as ordinary links -->
# <div id="_auto11"></div>
# 
# $$
# \begin{equation}
#     C(\boldsymbol{X}, \boldsymbol{\beta}; \lambda) = (\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y})^T(\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y}) + \lambda \boldsymbol{\beta}^T\boldsymbol{\beta}.
# \label{_auto11} \tag{11}
# \end{equation}
# $$

# In[26]:


_lambda = 0.1
clf_ridge = skl.Ridge(alpha=_lambda).fit(X_train, y_train)
J_ridge_sk = clf_ridge.coef_.reshape(L, L)
fig = plt.figure(figsize=(20, 14))
im = plt.imshow(J_ridge_sk, **cmap_args)
plt.title("Ridge from Scikit-learn", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb = fig.colorbar(im)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)

plt.show()


# In the **Least Absolute Shrinkage and Selection Operator** (LASSO)-method we get a third cost function.

# <!-- Equation labels as ordinary links -->
# <div id="_auto12"></div>
# 
# $$
# \begin{equation}
#     C(\boldsymbol{X}, \boldsymbol{\beta}; \lambda) = (\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y})^T(\boldsymbol{X}\boldsymbol{\beta} - \boldsymbol{y}) + \lambda \sqrt{\boldsymbol{\beta}^T\boldsymbol{\beta}}.
# \label{_auto12} \tag{12}
# \end{equation}
# $$

# Finding the extremal point of this cost function is not so straight-forward as in least squares and ridge. We will therefore rely solely on the function ``Lasso`` from **Scikit-Learn**.

# In[27]:


clf_lasso = skl.Lasso(alpha=_lambda).fit(X_train, y_train)
J_lasso_sk = clf_lasso.coef_.reshape(L, L)
fig = plt.figure(figsize=(20, 14))
im = plt.imshow(J_lasso_sk, **cmap_args)
plt.title("Lasso from Scikit-learn", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cb = fig.colorbar(im)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=18)

plt.show()


# It is quite striking how LASSO breaks the symmetry of the coupling
# constant as opposed to ridge and OLS. We get a sparse solution with
# $J_{j, j + 1} = -1$.
# 
# We see how the different models perform for a different set of values for $\lambda$.

# In[28]:


lambdas = np.logspace(-4, 5, 10)

train_errors = {
    "ols_sk": np.zeros(lambdas.size),
    "ridge_sk": np.zeros(lambdas.size),
    "lasso_sk": np.zeros(lambdas.size)
}

test_errors = {
    "ols_sk": np.zeros(lambdas.size),
    "ridge_sk": np.zeros(lambdas.size),
    "lasso_sk": np.zeros(lambdas.size)
}

plot_counter = 1

fig = plt.figure(figsize=(32, 54))

for i, _lambda in enumerate(tqdm.tqdm(lambdas)):
    for key, method in zip(
        ["ols_sk", "ridge_sk", "lasso_sk"],
        [skl.LinearRegression(), skl.Ridge(alpha=_lambda), skl.Lasso(alpha=_lambda)]
    ):
        method = method.fit(X_train, y_train)

        train_errors[key][i] = method.score(X_train, y_train)
        test_errors[key][i] = method.score(X_test, y_test)

        omega = method.coef_.reshape(L, L)

        plt.subplot(10, 5, plot_counter)
        plt.imshow(omega, **cmap_args)
        plt.title(r"%s, $\lambda = %.4f$" % (key, _lambda))
        plot_counter += 1

plt.show()


# We see that LASSO reaches a good solution for low
# values of $\lambda$, but will "wither" when we increase $\lambda$ too
# much. Ridge is more stable over a larger range of values for
# $\lambda$, but eventually also fades away.
# 
# To determine which value of $\lambda$ is best we plot the accuracy of
# the models when predicting the training and the testing set. We expect
# the accuracy of the training set to be quite good, but if the accuracy
# of the testing set is much lower this tells us that we might be
# subject to an overfit model. The ideal scenario is an accuracy on the
# testing set that is close to the accuracy of the training set.

# In[29]:


fig = plt.figure(figsize=(20, 14))

colors = {
    "ols_sk": "r",
    "ridge_sk": "y",
    "lasso_sk": "c"
}

for key in train_errors:
    plt.semilogx(
        lambdas,
        train_errors[key],
        colors[key],
        label="Train {0}".format(key),
        linewidth=4.0
    )

for key in test_errors:
    plt.semilogx(
        lambdas,
        test_errors[key],
        colors[key] + "--",
        label="Test {0}".format(key),
        linewidth=4.0
    )
plt.legend(loc="best", fontsize=18)
plt.xlabel(r"$\lambda$", fontsize=18)
plt.ylabel(r"$R^2$", fontsize=18)
plt.tick_params(labelsize=18)
plt.show()


# From the above figure we can see that LASSO with $\lambda = 10^{-2}$
# achieves a very good accuracy on the test set. This by far surpasses the
# other models for all values of $\lambda$.

# ## Exercises and Projects
# 
# The main aim of this project is to study in more detail various
# regression methods, including the Ordinary Least Squares (OLS) method,
# The total score is **100** points. Each subtask has its own final score.
# 
# We will first study how to fit polynomials to a specific
# two-dimensional function called [Franke's
# function](http://www.dtic.mil/dtic/tr/fulltext/u2/a081688.pdf).  This
# is a function which has been widely used when testing various
# interpolation and fitting algorithms. Furthermore, after having
# established the model and the method, we will employ resamling
# techniques such as cross-validation and/or bootstrap in order to perform a
# proper assessment of our models. We will also study in detail the
# so-called Bias-Variance trade off.
# 
# The Franke function, which is a weighted sum of four exponentials  reads as follows

# $$
# \begin{align*}
# f(x,y) &= \frac{3}{4}\exp{\left(-\frac{(9x-2)^2}{4} - \frac{(9y-2)^2}{4}\right)}+\frac{3}{4}\exp{\left(-\frac{(9x+1)^2}{49}- \frac{(9y+1)}{10}\right)} \\
# &+\frac{1}{2}\exp{\left(-\frac{(9x-7)^2}{4} - \frac{(9y-3)^2}{4}\right)} -\frac{1}{5}\exp{\left(-(9x-4)^2 - (9y-7)^2\right) }.
# \end{align*}
# $$

# The function will be defined for $x,y\in [0,1]$.  Our first step will
# be to perform an OLS regression analysis of this function, trying out
# a polynomial fit with an $x$ and $y$ dependence of the form $[x, y,
# x^2, y^2, xy, \dots]$. We will also include bootstrap first as
# a resampling technique.  After that we will include the cross-validation technique. As in homeworks 1 and 2, we can use a uniform
# distribution to set up the arrays of values for $x$ and $y$, or as in
# the example below just a set of fixed 
# values for $x$ and $y$ with a given step
# size.  We will fit a
# function (for example a polynomial) of $x$ and $y$.  Thereafter we
# will repeat much of the same procedure using the Ridge and Lasso
# regression methods, introducing thus a dependence on the bias
# (penalty) $\lambda$.
# 
# Finally we are going to use (real) digital terrain data and try to
# reproduce these data using the same methods. We will also try to go
# beyond the second-order polynomials metioned above and explore 
# which polynomial fits the data best.
# 
# The Python code for the Franke function is included here (it performs also a three-dimensional plot of it)

# In[30]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


z = FrankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# ### Exercise: Ordinary Least Square (OLS) on the Franke function
# 
# We will generate our own dataset for a function
# $\mathrm{FrankeFunction}(x,y)$ with $x,y \in [0,1]$. The function
# $f(x,y)$ is the Franke function. You should explore also the addition
# of an added stochastic noise to this function using the normal
# distribution $N(0,1)$.
# 
# *Write your own code* (using either a matrix inversion or a singular
# value decomposition from e.g., **numpy** ) or use your code from
# homeworks 1 and 2 and perform a standard least square regression
# analysis using polynomials in $x$ and $y$ up to fifth order. Find the
# [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) of the parameters (estimators) $\beta$ by computing their
# variances, evaluate the Mean Squared error (MSE)

# $$
# MSE(\hat{y},\hat{\tilde{y}}) = \frac{1}{n}
# \sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2,
# $$

# and the $R^2$ score function.  If $\tilde{\hat{y}}_i$ is the predicted
# value of the $i-th$ sample and $y_i$ is the corresponding true value,
# then the score $R^2$ is defined as

# $$
# R^2(\hat{y}, \tilde{\hat{y}}) = 1 - \frac{\sum_{i=0}^{n - 1} (y_i - \tilde{y}_i)^2}{\sum_{i=0}^{n - 1} (y_i - \bar{y})^2},
# $$

# where we have defined the mean value  of $\hat{y}$ as

# $$
# \bar{y} =  \frac{1}{n} \sum_{i=0}^{n - 1} y_i.
# $$

# Your code has to include a scaling of the data (for example by
# subtracting the mean value), and
# a split of the data in training and test data. For this exercise you can
# either write your own code or use for example the function for
# splitting training data provided by the library **Scikit-Learn** (make
# sure you have installed it).  This function is called
# $train\_test\_split$.  **You should present a critical discussion of why and how you have scaled or not scaled the data**.
# 
# It is normal in essentially all Machine Learning studies to split the
# data in a training set and a test set (eventually  also an additional
# validation set).  There
# is no explicit recipe for how much data should be included as training
# data and say test data.  An accepted rule of thumb is to use
# approximately $2/3$ to $4/5$ of the data as training data.
# 
# You can easily reuse the solutions to your exercises from week 35 and week 36.

# ### Exercise: Bias-variance trade-off and resampling techniques
# 
# Our aim here is to study the bias-variance trade-off by implementing the **bootstrap** resampling technique.
# 
# With a code which does OLS and includes resampling techniques, 
# we will now discuss the bias-variance trade-off in the context of
# continuous predictions such as regression. However, many of the
# intuitions and ideas discussed here also carry over to classification
# tasks and basically all Machine Learning algorithms. 
# 
# Before you perform an analysis of the bias-variance trade-off on your test data, make
# first a figure similar to Fig. 2.11 of Hastie, Tibshirani, and
# Friedman. Figure 2.11 of this reference displays only the test and training MSEs. The test MSE can be used to 
# indicate possible regions of low/high bias and variance. You will most likely not get an
# equally smooth curve!
# 
# With this result we move on to the bias-variance trade-off analysis.
# 
# Consider a
# dataset $\mathcal{L}$ consisting of the data
# $\mathbf{X}_\mathcal{L}=\{(y_j, \boldsymbol{x}_j), j=0\ldots n-1\}$.
# 
# Let us assume that the true data is generated from a noisy model

# $$
# \boldsymbol{y}=f(\boldsymbol{x}) + \boldsymbol{\epsilon}.
# $$

# Here $\epsilon$ is normally distributed with mean zero and standard
# deviation $\sigma^2$.
# 
# In our derivation of the ordinary least squares method we defined then
# an approximation to the function $f$ in terms of the parameters
# $\boldsymbol{\beta}$ and the design matrix $\boldsymbol{X}$ which embody our model,
# that is $\boldsymbol{\tilde{y}}=\boldsymbol{X}\boldsymbol{\beta}$.
# 
# The parameters $\boldsymbol{\beta}$ are in turn found by optimizing the means
# squared error via the so-called cost function

# $$
# C(\boldsymbol{X},\boldsymbol{\beta}) =\frac{1}{n}\sum_{i=0}^{n-1}(y_i-\tilde{y}_i)^2=\mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right].
# $$

# Here the expected value $\mathbb{E}$ is the sample value. 
# 
# Show that you can rewrite  this as

# $$
# \mathbb{E}\left[(\boldsymbol{y}-\boldsymbol{\tilde{y}})^2\right]=\frac{1}{n}\sum_i(f_i-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2+\frac{1}{n}\sum_i(\tilde{y}_i-\mathbb{E}\left[\boldsymbol{\tilde{y}}\right])^2+\sigma^2.
# $$

# Explain what the terms mean, which one is the bias and which one is
# the variance and discuss their interpretations.
# 
# Perform then a bias-variance analysis of the Franke function by
# studying the MSE value as function of the complexity of your model.
# 
# Discuss the bias and variance trade-off as function
# of your model complexity (the degree of the polynomial) and the number
# of data points, and possibly also your training and test data using the **bootstrap** resampling method.
# 
# Note also that when you calculate the bias, in all applications you don't know the function values $f_i$. You would hence replace them with the actual data points $y_i$.

# ### Exercise:  Cross-validation as resampling techniques, adding more complexity
# 
# The aim here is to write your own code for another widely popular
# resampling technique, the so-called cross-validation method.  Again,
# before you start with cross-validation approach, you should scale your
# data.
# 
# Implement the $k$-fold cross-validation algorithm (write your own
# code) and evaluate again the MSE function resulting
# from the test folds. You can compare your own code with that from
# **Scikit-Learn** if needed. 
# 
# Compare the MSE you get from your cross-validation code with the one
# you got from your **bootstrap** code. Comment your results. Try $5-10$
# folds.  You can also compare your own cross-validation code with the
# one provided by **Scikit-Learn**.

# ### Exercise: Ridge Regression on the Franke function  with resampling
# 
# Write your own code for the Ridge method, either using matrix
# inversion or the singular value decomposition as done in the previous
# exercise. Perform the same bootstrap analysis as in the
# Exercise 2  (for the same polynomials) and the cross-validation  in exercise 3 but now for different values of $\lambda$. Compare and
# analyze your results with those obtained in exercises 1-3. Study the
# dependence on $\lambda$.
# 
# Study also the bias-variance trade-off as function of various values of
# the parameter $\lambda$. For the bias-variance trade-off, use the **bootstrap** resampling method. Comment your results.

# ### Exercise: Lasso Regression on the Franke function  with resampling
# 
# This exercise is essentially a repeat of the previous two ones, but now
# with Lasso regression. Write either your own code (difficult and optional) or, in this case,
# you can also use the functionalities of **Scikit-Learn** (recommended). 
# Give a
# critical discussion of the three methods and a judgement of which
# model fits the data best.  Perform here as well an analysis of the bias-variance trade-off using the **bootstrap** resampling technique and an analysis of the mean squared error using cross-validation.

# ### Exercise: Analysis of real data
# 
# With our codes functioning and having been tested properly on a
# simpler function we are now ready to look at real data. We will
# essentially repeat in this exercise what was done in exercises 1-5. However, we
# need first to download the data and prepare properly the inputs to our
# codes.  We are going to download digital terrain data from the website
# <https://earthexplorer.usgs.gov/>,
# 
# Or, if you prefer, we have placed selected datafiles at <https://github.com/CompPhysics/MachineLearning/tree/master/doc/Projects/2021/Project1/DataFiles>
# 
# In order to obtain data for a specific region, you need to register as
# a user (free) at this website and then decide upon which area you want
# to fetch the digital terrain data from.  In order to be able to read
# the data properly, you need to specify that the format should be **SRTM
# Arc-Second Global** and download the data as a **GeoTIF** file.  The
# files are then stored in *tif* format which can be imported into a
# Python program using

# In[31]:


scipy.misc.imread


# Here is a simple part of a Python code which reads and plots the data
# from such files

# In[32]:


"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('SRTM_data_Norway_1.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""


# If you should have problems in downloading the digital terrain data,
# we provide two examples under the data folder of project 1. One is
# from a region close to Stavanger in Norway and the other Møsvatn
# Austfjell, again in Norway.
# Feel free to produce your own terrain data.
# 
# Alternatively, if you would like to use another data set, feel free to do so. This could be data close to your reseach area or simply a data set you found interesting. See for example [kaggle.com](https://www.kaggle.com/datasets) for examples.
# 
# Our final part deals with the parameterization of your digital terrain
# data (or your own data).  We will apply all three methods for linear regression, the same type (or higher order) of polynomial
# approximation and cross-validation as resampling technique to evaluate which
# model fits the data best.
# 
# At the end, you should present a critical evaluation of your results
# and discuss the applicability of these regression methods to the type
# of data presented here (either the terrain data we propose or other data sets).
