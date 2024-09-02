#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html statistics.do.txt  -->

# # Elements of Probability Theory and Statistical Data Analysis

# ## Domains and probabilities
# 
# Consider the following simple example, namely the tossing of two dice, resulting in  the following possible values

# $$
# \{2,3,4,5,6,7,8,9,10,11,12\}.
# $$

# These values are called the *domain*. 
# To this domain we have the corresponding *probabilities*

# $$
# \{1/36,2/36/,3/36,4/36,5/36,6/36,5/36,4/36,3/36,2/36,1/36\}.
# $$

# The numbers in the domain are the outcomes of the physical process of tossing say two dice.
# We cannot tell beforehand whether the outcome is 3 or 5 or any other number in this domain.
# This defines the randomness of the outcome, or unexpectedness or any other synonimous word which
# encompasses the uncertitude of the final outcome. 
# 
# The only thing we can tell beforehand
# is that say the outcome 2 has a certain probability.  
# If our favorite hobby is to  spend an hour every evening throwing dice and 
# registering the sequence of outcomes, we will note that the numbers in the above domain

# $$
# \{2,3,4,5,6,7,8,9,10,11,12\},
# $$

# appear in a random order. After 11 throws the results may look like

# $$
# \{10,8,6,3,6,9,11,8,12,4,5\}.
# $$

# **Random variables are characterized by a domain which contains all possible values that the random value may take. This domain has a corresponding probability distribution function(PDF)**.

# ### Stochastic variables and the main concepts, the discrete case
# 
# There are two main concepts associated with a stochastic variable. The
# *domain* is the set $\mathbb D = \{x\}$ of all accessible values
# the variable can assume, so that $X \in \mathbb D$. An example of a
# discrete domain is the set of six different numbers that we may get by
# throwing of a dice, $x\in\{1,\,2,\,3,\,4,\,5,\,6\}$.
# 
# The *probability distribution function (PDF)* is a function
# $p(x)$ on the domain which, in the discrete case, gives us the
# probability or relative frequency with which these values of $X$
# occur

# $$
# p(x) = \mathrm{Prob}(X=x).
# $$

# In the continuous case, the PDF does not directly depict the
# actual probability. Instead we define the probability for the
# stochastic variable to assume any value on an infinitesimal interval
# around $x$ to be $p(x)dx$. The continuous function $p(x)$ then gives us
# the *density* of the probability rather than the probability
# itself. The probability for a stochastic variable to assume any value
# on a non-infinitesimal interval $[a,\,b]$ is then just the integral

# $$
# \mathrm{Prob}(a\leq X\leq b) = \int_a^b p(x)dx.
# $$

# Qualitatively speaking, a stochastic variable represents the values of
# numbers chosen as if by chance from some specified PDF so that the
# selection of a large set of these numbers reproduces this PDF.
# 
# Of interest to us is the *cumulative probability
# distribution function* (**CDF**), $P(x)$, which is just the probability
# for a stochastic variable $X$ to assume any value less than $x$

# $$
# P(x)=\mathrm{Prob(}X\leq x\mathrm{)} =
# \int_{-\infty}^x p(x^{\prime})dx^{\prime}.
# $$

# The relation between a CDF and its corresponding PDF is then

# $$
# p(x) = \frac{d}{dx}P(x).
# $$

# ### Properties of PDFs
# 
# There are two properties that all PDFs must satisfy. The first one is
# positivity (assuming that the PDF is normalized)

# $$
# 0 \leq p(x) \leq 1.
# $$

# Naturally, it would be nonsensical for any of the values of the domain
# to occur with a probability greater than $1$ or less than $0$. Also,
# the PDF must be normalized. That is, all the probabilities must add up
# to unity.  The probability of "anything" to happen is always unity. For
# both discrete and continuous PDFs, this condition is

# $$
# \begin{align*}
# \sum_{x_i\in\mathbb D} p(x_i) & =  1,\\
# \int_{x\in\mathbb D} p(x)\,dx & =  1.
# \end{align*}
# $$

# The first one
# is the most basic PDF; namely the uniform distribution

# <!-- Equation labels as ordinary links -->
# <div id="eq:unifromPDF"></div>
# 
# $$
# \begin{equation}
# p(x) = \frac{1}{b-a}\theta(x-a)\theta(b-x).
# \label{eq:unifromPDF} \tag{1}
# \end{equation}
# $$

# For $a=0$ and $b=1$ we have

# $$
# \begin{array}{ll}
# p(x)dx = dx & \in [0,1].
# \end{array}
# $$

# The latter distribution is used to generate random numbers. For other PDFs, one needs normally a mapping from this distribution to say for example the exponential distribution. 
# 
# The second one is the Gaussian Distribution

# $$
# p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp{(-\frac{(x-\mu)^2}{2\sigma^2})},
# $$

# with mean value $\mu$ and standard deviation $\sigma$. If $\mu=0$ and $\sigma=1$, it is normally called the **standard normal distribution**

# $$
# p(x) = \frac{1}{\sqrt{2\pi}} \exp{(-\frac{x^2}{2})},
# $$

# The following simple Python code plots the above distribution for different values of $\mu$ and $\sigma$.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from math import acos, exp, sqrt
from  matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.units as units
import matplotlib.ticker as ticker
rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Gaussian distribution']})
font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }
pi = acos(-1.0)
mu0 = 0.0
sigma0 = 1.0
mu1= 1.0
sigma1 = 2.0
mu2 = 2.0
sigma2 = 4.0

x = np.linspace(-20.0, 20.0)
v0 = np.exp(-(x*x-2*x*mu0+mu0*mu0)/(2*sigma0*sigma0))/sqrt(2*pi*sigma0*sigma0)
v1 = np.exp(-(x*x-2*x*mu1+mu1*mu1)/(2*sigma1*sigma1))/sqrt(2*pi*sigma1*sigma1)
v2 = np.exp(-(x*x-2*x*mu2+mu2*mu2)/(2*sigma2*sigma2))/sqrt(2*pi*sigma2*sigma2)
plt.plot(x, v0, 'b-', x, v1, 'r-', x, v2, 'g-')
plt.title(r'{\bf Gaussian distributions}', fontsize=20)
plt.text(-19, 0.3, r'Parameters: $\mu = 0$, $\sigma = 1$', fontdict=font)
plt.text(-19, 0.18, r'Parameters: $\mu = 1$, $\sigma = 2$', fontdict=font)
plt.text(-19, 0.08, r'Parameters: $\mu = 2$, $\sigma = 4$', fontdict=font)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$p(x)$ [MeV]',fontsize=20)

# Tweak spacing to prevent clipping of ylabel                                                                       
plt.subplots_adjust(left=0.15)
plt.savefig('gaussian.pdf', format='pdf')
plt.show()


# Another important distribution in science is the exponential distribution

# $$
# p(x) = \alpha\exp{-(\alpha x)}.
# $$

# ### Expectation values
# 
# Let $h(x)$ be an arbitrary continuous function on the domain of the stochastic
# variable $X$ whose PDF is $p(x)$. We define the *expectation value*
# of $h$ with respect to $p$ as follows

# <!-- Equation labels as ordinary links -->
# <div id="eq:expectation_value_of_h_wrt_p"></div>
# 
# $$
# \begin{equation}
# \langle h \rangle_X \equiv \int\! h(x)p(x)\,dx
# \label{eq:expectation_value_of_h_wrt_p} \tag{2}
# \end{equation}
# $$

# Whenever the PDF is known implicitly, like in this case, we will drop
# the index $X$ for clarity.  
# A particularly useful class of special expectation values are the
# *moments*. The $n$-th moment of the PDF $p$ is defined as
# follows

# $$
# \langle x^n \rangle \equiv \int\! x^n p(x)\,dx
# $$

# The zero-th moment $\langle 1\rangle$ is just the normalization condition of
# $p$. The first moment, $\langle x\rangle$, is called the *mean* of $p$
# and often denoted by the letter $\mu$

# $$
# \langle x\rangle  = \mu \equiv \int x p(x)dx,
# $$

# for a continuous distribution and

# $$
# \langle x\rangle  = \mu \equiv \sum_{i=1}^N x_i p(x_i),
# $$

# for a discrete distribution. 
# Qualitatively it represents the centroid or the average value of the
# PDF and is therefore simply called the expectation value of $p(x)$.
# 
# A special version of the moments is the set of *central moments*, the n-th central moment defined as

# $$
# \langle (x-\langle x\rangle )^n\rangle  \equiv \int\! (x-\langle x\rangle)^n p(x)\,dx
# $$

# The zero-th and first central moments are both trivial, equal $1$ and
# $0$, respectively. But the second central moment, known as the
# *variance* of $p$, is of particular interest. For the stochastic
# variable $X$, the variance is denoted as $\sigma^2_X$ or $\mathrm{Var}(X)$

# $$
# \begin{align*}
# \sigma^2_X &=\mathrm{Var}(X) =  \langle (x-\langle x\rangle)^2\rangle  =
# \int (x-\langle x\rangle)^2 p(x)dx\\
# & =  \int\left(x^2 - 2 x \langle x\rangle^{2} +\langle x\rangle^2\right)p(x)dx\\
# & =  \langle x^2\rangle - 2 \langle x\rangle\langle x\rangle + \langle x\rangle^2\\
# & =  \langle x^2 \rangle - \langle x\rangle^2
# \end{align*}
# $$

# The square root of the variance, $\sigma =\sqrt{\langle (x-\langle x\rangle)^2\rangle}$ is called the 
# **standard deviation** of $p$. It is the RMS (root-mean-square)
# value of the deviation of the PDF from its mean value, interpreted
# qualitatively as the "spread" of $p$ around its mean.

# ### Probability Distribution Functions
# 
# The following table collects properties of probability distribution functions.
# In our notation we reserve the label $p(x)$ for the probability of a certain event,
# while $P(x)$ is the cumulative probability. 
# 
# <table class="dotable" border="1">
# <thead>
# <tr><th align="center">             </th> <th align="center">               Discrete PDF               </th> <th align="center">           Continuous PDF           </th> </tr>
# </thead>
# <tbody>
# <tr><td align="left">   Domain           </td> <td align="center">   $\left\{x_1, x_2, x_3, \dots, x_N\right\}$    </td> <td align="center">   $[a,b]$                                 </td> </tr>
# <tr><td align="left">   Probability      </td> <td align="center">   $p(x_i)$                                      </td> <td align="center">   $p(x)dx$                                </td> </tr>
# <tr><td align="left">   Cumulative       </td> <td align="center">   $P_i=\sum_{l=1}^ip(x_l)$                      </td> <td align="center">   $P(x)=\int_a^xp(t)dt$                   </td> </tr>
# <tr><td align="left">   Positivity       </td> <td align="center">   $0 \le p(x_i) \le 1$                          </td> <td align="center">   $p(x) \ge 0$                            </td> </tr>
# <tr><td align="left">   Positivity       </td> <td align="center">   $0 \le P_i \le 1$                             </td> <td align="center">   $0 \le P(x) \le 1$                      </td> </tr>
# <tr><td align="left">   Monotonic        </td> <td align="center">   $P_i \ge P_j$ if $x_i \ge x_j$                </td> <td align="center">   $P(x_i) \ge P(x_j)$ if $x_i \ge x_j$    </td> </tr>
# <tr><td align="left">   Normalization    </td> <td align="center">   $P_N=1$                                       </td> <td align="center">   $P(b)=1$                                </td> </tr>
# </tbody>
# </table>
# 
# With a PDF we can compute expectation values of selected quantities such as

# $$
# \langle x^k\rangle=\sum_{i=1}^{N}x_i^kp(x_i),
# $$

# if we have a discrete PDF or

# $$
# \langle x^k\rangle=\int_a^b x^kp(x)dx,
# $$

# in the case of a continuous PDF. We have already defined the mean value $\mu$
# and the variance $\sigma^2$. 
# 
# There are at least three PDFs which one may encounter. These are the
# 
# **Uniform distribution**

# $$
# p(x)=\frac{1}{b-a}\Theta(x-a)\Theta(b-x),
# $$

# yielding probabilities different from zero in the interval $[a,b]$.
# 
# **The exponential distribution**

# $$
# p(x)=\alpha \exp{(-\alpha x)},
# $$

# yielding probabilities different from zero in the interval $[0,\infty)$ and with mean value

# $$
# \mu = \int_0^{\infty}xp(x)dx=\int_0^{\infty}x\alpha \exp{(-\alpha x)}dx=\frac{1}{\alpha},
# $$

# with variance

# $$
# \sigma^2=\int_0^{\infty}x^2p(x)dx-\mu^2 = \frac{1}{\alpha^2}.
# $$

# Finally, we have the so-called univariate normal  distribution, or just the **normal distribution**

# $$
# p(x)=\frac{1}{b\sqrt{2\pi}}\exp{\left(-\frac{(x-a)^2}{2b^2}\right)}
# $$

# with probabilities different from zero in the interval $(-\infty,\infty)$.
# The integral $\int_{-\infty}^{\infty}\exp{\left(-(x^2\right)}dx$ appears in many calculations, its value
# is $\sqrt{\pi}$,  a result we will need when we compute the mean value and the variance.
# The mean value is

# $$
# \mu = \int_0^{\infty}xp(x)dx=\frac{1}{b\sqrt{2\pi}}\int_{-\infty}^{\infty}x \exp{\left(-\frac{(x-a)^2}{2b^2}\right)}dx,
# $$

# which becomes with a suitable change of variables

# $$
# \mu =\frac{1}{b\sqrt{2\pi}}\int_{-\infty}^{\infty}b\sqrt{2}(a+b\sqrt{2}y)\exp{-y^2}dy=a.
# $$

# Similarly, the variance becomes

# $$
# \sigma^2 = \frac{1}{b\sqrt{2\pi}}\int_{-\infty}^{\infty}(x-\mu)^2 \exp{\left(-\frac{(x-a)^2}{2b^2}\right)}dx,
# $$

# and inserting the mean value and performing a variable change we obtain

# $$
# \sigma^2 = \frac{1}{b\sqrt{2\pi}}\int_{-\infty}^{\infty}b\sqrt{2}(b\sqrt{2}y)^2\exp{\left(-y^2\right)}dy=
# \frac{2b^2}{\sqrt{\pi}}\int_{-\infty}^{\infty}y^2\exp{\left(-y^2\right)}dy,
# $$

# and performing a final integration by parts we obtain the well-known result $\sigma^2=b^2$.
# It is useful to introduce the standard normal distribution as well, defined by $\mu=a=0$, viz. a distribution
# centered around zero and with a variance $\sigma^2=1$, leading to

# <!-- Equation labels as ordinary links -->
# <div id="_auto1"></div>
# 
# $$
# \begin{equation}
#    p(x)=\frac{1}{\sqrt{2\pi}}\exp{\left(-\frac{x^2}{2}\right)}.
# \label{_auto1} \tag{3}
# \end{equation}
# $$

# The exponential and uniform distributions have simple cumulative functions,
# whereas the normal distribution does not, being proportional to the so-called
# error function $erf(x)$, given by

# $$
# P(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x\exp{\left(-\frac{t^2}{2}\right)}dt,
# $$

# which is difficult to evaluate in a quick way. 
# 
# Some other PDFs which one encounters often in the natural sciences are the binomial distribution

# $$
# p(x) = \left(\begin{array}{c} n \\ x\end{array}\right)y^x(1-y)^{n-x} \hspace{0.5cm}x=0,1,\dots,n,
# $$

# where $y$ is the probability for a specific event, such as the tossing of a coin or moving left or right
# in case of a random walker. Note that $x$ is a discrete stochastic variable. 
# 
# The sequence of binomial trials is characterized by the following definitions
# 
#   * Every experiment is thought to consist of $N$ independent trials.
# 
#   * In every independent trial one registers if a specific situation happens or not, such as the  jump to the left or right of a random walker.
# 
#   * The probability for every outcome in a single trial has the same value, for example the outcome of tossing (either heads or tails) a coin is always $1/2$.
# 
# In order to compute the mean and variance we need to recall Newton's binomial
# formula

# $$
# (a+b)^m=\sum_{n=0}^m \left(\begin{array}{c} m \\ n\end{array}\right)a^nb^{m-n},
# $$

# which can be used to show that

# $$
# \sum_{x=0}^n\left(\begin{array}{c} n \\ x\end{array}\right)y^x(1-y)^{n-x} = (y+1-y)^n = 1,
# $$

# the PDF is normalized to one. 
# The mean value is

# $$
# \mu = \sum_{x=0}^n x\left(\begin{array}{c} n \\ x\end{array}\right)y^x(1-y)^{n-x} =
# \sum_{x=0}^n x\frac{n!}{x!(n-x)!}y^x(1-y)^{n-x},
# $$

# resulting in

# $$
# \mu = 
# \sum_{x=0}^n x\frac{(n-1)!}{(x-1)!(n-1-(x-1))!}y^{x-1}(1-y)^{n-1-(x-1)},
# $$

# which we rewrite as

# $$
# \mu=ny\sum_{\nu=0}^n\left(\begin{array}{c} n-1 \\ \nu\end{array}\right)y^{\nu}(1-y)^{n-1-\nu} =ny(y+1-y)^{n-1}=ny.
# $$

# The variance is slightly trickier to get. It reads $\sigma^2=ny(1-y)$. 
# 
# Another important distribution with discrete stochastic variables $x$ is  
# the Poisson model, which resembles the exponential distribution and reads

# $$
# p(x) = \frac{\lambda^x}{x!} e^{-\lambda} \hspace{0.5cm}x=0,1,\dots,;\lambda > 0.
# $$

# In this case both the mean value and the variance are easier to calculate,

# $$
# \mu = \sum_{x=0}^{\infty} x \frac{\lambda^x}{x!} e^{-\lambda} = \lambda e^{-\lambda}\sum_{x=1}^{\infty}
# \frac{\lambda^{x-1}}{(x-1)!}=\lambda,
# $$

# and the variance is $\sigma^2=\lambda$. 
# 
# An example of applications of the Poisson distribution could be the counting
# of the number of $\alpha$-particles emitted from a radioactive source in a given time interval.
# In the limit of $n\rightarrow \infty$ and for small probabilities $y$, the binomial distribution
# approaches the Poisson distribution. Setting $\lambda = ny$, with $y$ the probability for an event in
# the binomial distribution we can show that

# $$
# \lim_{n\rightarrow \infty}\left(\begin{array}{c} n \\ x\end{array}\right)y^x(1-y)^{n-x} e^{-\lambda}=\sum_{x=1}^{\infty}\frac{\lambda^x}{x!} e^{-\lambda}.
# $$

# ### Meet the  covariance!
# 
# An important quantity in a statistical analysis is the so-called covariance. 
# 
# Consider the set $\{X_i\}$ of $n$
# stochastic variables (not necessarily uncorrelated) with the
# multivariate PDF $P(x_1,\dots,x_n)$. The *covariance* of two
# of the stochastic variables, $X_i$ and $X_j$, is defined as follows

# <!-- Equation labels as ordinary links -->
# <div id="_auto2"></div>
# 
# $$
# \begin{equation}
# \mathrm{Cov}(X_i,\,X_j)  = \langle (x_i-\langle x_i\rangle)(x_j-\langle x_j\rangle)\rangle 
# \label{_auto2} \tag{4}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="eq:def_covariance"></div>
# 
# $$
# \begin{equation} 
# =\int\cdots\int (x_i-\langle x_i\rangle)(x_j-\langle x_j\rangle)P(x_1,\dots,x_n)\,dx_1\dots dx_n,
# \label{eq:def_covariance} \tag{5}
# \end{equation}
# $$

# with

# $$
# \langle x_i\rangle =
# \int\cdots\int x_i P(x_1,\dots,x_n)\,dx_1\dots dx_n.
# $$

# If we consider the above covariance as a matrix

# $$
# C_{ij} =\mathrm{Cov}(X_i,\,X_j),
# $$

# then the diagonal elements are just the familiar
# variances, $C_{ii} = \mathrm{Cov}(X_i,\,X_i) = \mathrm{Var}(X_i)$. It turns out that
# all the off-diagonal elements are zero if the stochastic variables are
# uncorrelated.

# In[2]:


# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

def covariance(x, y, n):
    sum = 0.0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    for i in range(0, n):
        sum += (x[(i)]-mean_x)*(y[i]-mean_y)
    return  sum/n

n = 10

x=np.random.normal(size=n)
y = 4+3*x+np.random.normal(size=n)
covxy = covariance(x,y,n)
print(covxy)
z = np.vstack((x, y))
c = np.cov(z.T)

print(c)


# Consider the stochastic variables $X_i$ and $X_j$, ($i\neq j$). We have

# $$
# \begin{align*}
# Cov(X_i,\,X_j) &= \langle (x_i-\langle x_i\rangle)(x_j-\langle x_j\rangle)\rangle\\
# &=\langle x_i x_j - x_i\langle x_j\rangle - \langle x_i\rangle x_j + \langle x_i\rangle\langle x_j\rangle\rangle\\
# &=\langle x_i x_j\rangle - \langle x_i\langle x_j\rangle\rangle - \langle \langle x_i\rangle x_j \rangle +
# \langle \langle x_i\rangle\langle x_j\rangle\rangle \\
# &=\langle x_i x_j\rangle - \langle x_i\rangle\langle x_j\rangle - \langle x_i\rangle\langle x_j\rangle +
# \langle x_i\rangle\langle x_j\rangle \\
# &=\langle x_i x_j\rangle - \langle x_i\rangle\langle x_j\rangle
# \end{align*}
# $$

# If $X_i$ and $X_j$ are independent (assuming $i \neq j$), we have that

# $$
# \langle x_i x_j\rangle = \langle x_i\rangle\langle x_j\rangle,
# $$

# leading to

# $$
# Cov(X_i, X_j) = 0 \hspace{0.1cm} (i\neq j).
# $$

# Now that we have constructed an idealized mathematical framework, let
# us try to apply it to empirical observations. Examples of relevant
# physical phenomena may be spontaneous decays of nuclei, or a purely
# mathematical set of numbers produced by some deterministic
# mechanism. It is the latter we will deal with, using so-called pseudo-random
# number generators.  In general our observations will contain only a limited set of
# observables. We remind the reader that
# a *stochastic process* is a process that produces sequentially a
# chain of values

# $$
# \{x_1, x_2,\dots\,x_k,\dots\}.
# $$

# We will call these
# values our *measurements* and the entire set as our measured
# *sample*.  The action of measuring all the elements of a sample
# we will call a stochastic *experiment* (since, operationally,
# they are often associated with results of empirical observation of
# some physical or mathematical phenomena; precisely an experiment). We
# assume that these values are distributed according to some 
# PDF $p_X^{\phantom X}(x)$, where $X$ is just the formal symbol for the
# stochastic variable whose PDF is $p_X^{\phantom X}(x)$. Instead of
# trying to determine the full distribution $p$ we are often only
# interested in finding the few lowest moments, like the mean
# $\mu_X^{\phantom X}$ and the variance $\sigma_X^{\phantom X}$.
# 
# In practical situations however, a sample is always of finite size. Let that
# size be $n$. The expectation value of a sample $\alpha$, the **sample mean**, is then defined as follows

# $$
# \langle x_{\alpha} \rangle \equiv \frac{1}{n}\sum_{k=1}^n x_{\alpha,k}.
# $$

# The *sample variance* is:

# $$
# \mathrm{Var}(x) \equiv \frac{1}{n}\sum_{k=1}^n (x_{\alpha,k} - \langle x_{\alpha} \rangle)^2,
# $$

# with its square root being the *standard deviation of the sample*. 
# 
# You can think of the above observables as a set of quantities which define
# a given experiment. This experiment is then repeated several times, say $m$ times.
# The total average is then

# <!-- Equation labels as ordinary links -->
# <div id="eq:exptmean"></div>
# 
# $$
# \begin{equation}
# \langle X_m \rangle= \frac{1}{m}\sum_{\alpha=1}^mx_{\alpha}=\frac{1}{mn}\sum_{\alpha, k} x_{\alpha,k},
# \label{eq:exptmean} \tag{6}
# \end{equation}
# $$

# where the last sums end at $m$ and $n$.
# The total variance is

# $$
# \sigma^2_m= \frac{1}{mn^2}\sum_{\alpha=1}^m(\langle x_{\alpha} \rangle-\langle X_m \rangle)^2,
# $$

# which we rewrite as

# <!-- Equation labels as ordinary links -->
# <div id="eq:exptvariance"></div>
# 
# $$
# \begin{equation}
# \sigma^2_m=\frac{1}{m}\sum_{\alpha=1}^m\sum_{kl=1}^n (x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,l}-\langle X_m \rangle).
# \label{eq:exptvariance} \tag{7}
# \end{equation}
# $$

# We define also the sample variance $\sigma^2$ of all $mn$ individual experiments as

# <!-- Equation labels as ordinary links -->
# <div id="eq:sampleexptvariance"></div>
# 
# $$
# \begin{equation}
# \sigma^2=\frac{1}{mn}\sum_{\alpha=1}^m\sum_{k=1}^n (x_{\alpha,k}-\langle X_m \rangle)^2.
# \label{eq:sampleexptvariance} \tag{8}
# \end{equation}
# $$

# These quantities, being known experimental values or the results from our calculations, 
# may differ, in some cases
# significantly,  from the similarly named
# exact values for the mean value $\mu_X$, the variance $\mathrm{Var}(X)$
# and the covariance $\mathrm{Cov}(X,Y)$.

# ### Numerical experiments and the covariance, central limit theorem
# 
# The central limit theorem states that the PDF $\tilde{p}(z)$ of
# the average of $m$ random values corresponding to a PDF $p(x)$ 
# is a normal distribution whose mean is the 
# mean value of the PDF $p(x)$ and whose variance is the variance
# of the PDF $p(x)$ divided by $m$, the number of values used to compute $z$.
# 
# The central limit theorem leads then to the well-known expression for the
# standard deviation, given by

# $$
# \sigma_m=
# \frac{\sigma}{\sqrt{m}}.
# $$

# In many cases the above estimate for the standard deviation, in particular if correlations are strong, may be too simplistic.  We need therefore a more precise defintion of the error and the variance in our results.
# 
# Our estimate of the true average $\mu_{X}$ is the sample mean $\langle X_m \rangle$

# $$
# \mu_{X}^{\phantom X} \approx X_m=\frac{1}{mn}\sum_{\alpha=1}^m\sum_{k=1}^n x_{\alpha,k}.
# $$

# We can then use Eq. ([7](#eq:exptvariance))

# $$
# \sigma^2_m=\frac{1}{mn^2}\sum_{\alpha=1}^m\sum_{kl=1}^n (x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,l}-\langle X_m \rangle),
# $$

# and rewrite it as

# $$
# \sigma^2_m=\frac{\sigma^2}{n}+\frac{2}{mn^2}\sum_{\alpha=1}^m\sum_{k<l}^n (x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,l}-\langle X_m \rangle),
# $$

# where the first term is the sample variance of all $mn$ experiments divided by $n$
# and the last term is nothing but the covariance which arises when $k\ne l$. 
# 
# Our estimate of the true average $\mu_{X}$ is the sample mean $\langle X_m \rangle$
# 
# If the 
# observables are uncorrelated, then the covariance is zero and we obtain a total variance
# which agrees with the central limit theorem. Correlations may often be present in our data set, resulting in a non-zero covariance.  The first term is normally called the uncorrelated 
# contribution.
# Computationally the uncorrelated first term is much easier to treat
# efficiently than the second.
# We just accumulate separately the values $x^2$ and $x$ for every
# measurement $x$ we receive. The correlation term, though, has to be
# calculated at the end of the experiment since we need all the
# measurements to calculate the cross terms. Therefore, all measurements
# have to be stored throughout the experiment.
# 
# Let us analyze the problem by splitting up the correlation term into
# partial sums of the form

# $$
# f_d = \frac{1}{nm}\sum_{\alpha=1}^m\sum_{k=1}^{n-d}(x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,k+d}-\langle X_m \rangle),
# $$

# The correlation term of the total variance can now be rewritten in terms of
# $f_d$

# $$
# \frac{2}{mn^2}\sum_{\alpha=1}^m\sum_{k<l}^n (x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,l}-\langle X_m \rangle)=
# \frac{2}{n}\sum_{d=1}^{n-1} f_d
# $$

# The value of $f_d$ reflects the correlation between measurements
# separated by the distance $d$ in the samples.  Notice that for
# $d=0$, $f$ is just the sample variance, $\sigma^2$. If we divide $f_d$
# by $\sigma^2$, we arrive at the so called **autocorrelation function**

# <!-- Equation labels as ordinary links -->
# <div id="eq:autocorrelformal"></div>
# 
# $$
# \begin{equation}
# \kappa_d = \frac{f_d}{\sigma^2}
# \label{eq:autocorrelformal} \tag{9}
# \end{equation}
# $$

# which gives us a useful measure of the correlation pair correlation
# starting always at $1$ for $d=0$.
# 
# The sample variance of the $mn$ experiments can now be
# written in terms of the autocorrelation function

# <!-- Equation labels as ordinary links -->
# <div id="eq:error_estimate_corr_time"></div>
# 
# $$
# \begin{equation}
# \sigma_m^2=\frac{\sigma^2}{n}+\frac{2}{n}\cdot\sigma^2\sum_{d=1}^{n-1}
# \frac{f_d}{\sigma^2}=\left(1+2\sum_{d=1}^{n-1}\kappa_d\right)\frac{1}{n}\sigma^2=\frac{\tau}{n}\cdot\sigma^2
# \label{eq:error_estimate_corr_time} \tag{10}
# \end{equation}
# $$

# and we see that $\sigma_m$ can be expressed in terms of the
# uncorrelated sample variance times a correction factor $\tau$ which
# accounts for the correlation between measurements. We call this
# correction factor the *autocorrelation time*

# <!-- Equation labels as ordinary links -->
# <div id="eq:autocorrelation_time"></div>
# 
# $$
# \begin{equation}
# \tau = 1+2\sum_{d=1}^{n-1}\kappa_d
# \label{eq:autocorrelation_time} \tag{11}
# \end{equation}
# $$

# <!-- It is closely related to the area under the graph of the -->
# <!-- autocorrelation function. -->
# For a correlation free experiment, $\tau$
# equals 1. 
# 
# From the point of view of
# Eq. ([10](#eq:error_estimate_corr_time)) we can interpret a sequential
# correlation as an effective reduction of the number of measurements by
# a factor $\tau$. The effective number of measurements becomes

# $$
# n_\mathrm{eff} = \frac{n}{\tau}
# $$

# To neglect the autocorrelation time $\tau$ will always cause our
# simple uncorrelated estimate of $\sigma_m^2\approx \sigma^2/n$ to
# be less than the true sample error. The estimate of the error will be
# too "good". On the other hand, the calculation of the full
# autocorrelation time poses an efficiency problem if the set of
# measurements is very large.  The solution to this problem is given by 
# more practically oriented methods like the blocking technique.
# <!-- add ref here to flybjerg -->

# In[3]:


# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

# Sample covariance, note the factor 1/(n-1)
def covariance(x, y, n):
    sum = 0.0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    for i in range(0, n):
        sum += (x[(i)]-mean_x)*(y[i]-mean_y)
    return  sum/(n-1.)

n = 100
x = np.random.normal(size=n)
print(np.mean(x))
y = 4+3*x+np.random.normal(size=n)
print(np.mean(y))
z = x**3+np.random.normal(size=n)
print(np.mean(z))
covxx = covariance(x,x,n)
covyy = covariance(y,y,n)
covzz = covariance(z,z,n)
covxy = covariance(x,y,n)
covxz = covariance(x,z,n)
covyz = covariance(y,z,n)
print(covxx,covyy, covzz)
print(covxy,covxz, covyz)
w = np.vstack((x, y, z))
#print(w)
c = np.cov(w)
print(c)
#eigen = np.zeros(n)
Eigvals, Eigvecs = np.linalg.eig(c)
print(Eigvals)


# ### Random Numbers
# 
# Uniform deviates are just random numbers that lie within a specified range
# (typically 0 to 1), with any one number in the range just as likely as any other. They
# are, in other words, what you probably think random numbers are. However,
# we want to distinguish uniform deviates from other sorts of random numbers, for
# example numbers drawn from a normal (Gaussian) distribution of specified mean
# and standard deviation. These other sorts of deviates are almost always generated by
# performing appropriate operations on one or more uniform deviates, as we will see
# in subsequent sections. So, a reliable source of random uniform deviates, the subject
# of this section, is an essential building block for any sort of stochastic modeling
# or Monte Carlo computer work.
# 
# A disclaimer is however appropriate. It should be fairly obvious that 
# something as deterministic as a computer cannot generate purely random numbers.
# 
# Numbers generated by any of the standard algorithms are in reality pseudo random
# numbers, hopefully abiding to the following criteria:
# 
#   * they produce a uniform distribution in the interval [0,1].
# 
#   * correlations between random numbers are negligible
# 
#   * the period before the same sequence of random numbers is repeated   is as large as possible and finally
# 
#   * the algorithm should be fast.
# 
#  The most common random number generators are based on so-called
# Linear congruential relations of the type

# $$
# N_i=(aN_{i-1}+c) \mathrm{MOD} (M),
# $$

# which yield a number in the interval [0,1] through

# $$
# x_i=N_i/M
# $$

# The number 
# $M$ is called the period and it should be as large as possible 
#  and 
# $N_0$ is the starting value, or seed. The function $\mathrm{MOD}$ means the remainder,
# that is if we were to evaluate $(13)\mathrm{MOD}(9)$, the outcome is the remainder
# of the division $13/9$, namely $4$.
# 
# The problem with such generators is that their outputs are periodic;
# they 
# will start to repeat themselves with a period that is at most $M$. If however
# the parameters $a$ and $c$ are badly chosen, the period may be even shorter.
# 
# Consider the following example

# $$
# N_i=(6N_{i-1}+7) \mathrm{MOD} (5),
# $$

# with a seed $N_0=2$. This generator produces the sequence
# $4,1,3,0,2,4,1,3,0,2,...\dots$, i.e., a sequence with period $5$.
# However, increasing $M$ may not guarantee a larger period as the following
# example shows

# $$
# N_i=(27N_{i-1}+11) \mathrm{MOD} (54),
# $$

# which still, with $N_0=2$, results in $11,38,11,38,11,38,\dots$, a period of
# just $2$.
# 
# Typical periods for the random generators provided in the program library 
# are of the order of $\sim 10^9$ or larger. Other random number generators which have
# become increasingly popular are so-called shift-register generators.
# In these generators each successive number depends on many preceding
# values (rather than the last values as in the linear congruential
# generator).
# For example, you could make a shift register generator whose $l$th 
# number is the sum of the $l-i$th and $l-j$th values with modulo $M$,

# $$
# N_l=(aN_{l-i}+cN_{l-j})\mathrm{MOD}(M).
# $$

# Such a generator again produces a sequence of pseudorandom numbers
# but this time with a period much larger than $M$.
# It is also possible to construct more elaborate algorithms by including
# more than two past terms in the sum of each iteration.
# One example is the generator of [Marsaglia and Zaman](http://dl.acm.org/citation.cfm?id=187154)
# which consists of two congruential relations

# <!-- Equation labels as ordinary links -->
# <div id="eq:mz1"></div>
# 
# $$
# \begin{equation}
#    N_l=(N_{l-3}-N_{l-1})\mathrm{MOD}(2^{31}-69),
# \label{eq:mz1} \tag{12}
# \end{equation}
# $$

# followed by

# <!-- Equation labels as ordinary links -->
# <div id="eq:mz2"></div>
# 
# $$
# \begin{equation}
#    N_l=(69069N_{l-1}+1013904243)\mathrm{MOD}(2^{32}),
# \label{eq:mz2} \tag{13}
# \end{equation}
# $$

# which according to the authors has a period larger than $2^{94}$.
# 
# Instead of  using modular addition, we could use the bitwise
# exclusive-OR ($\oplus$) operation so that

# $$
# N_l=(N_{l-i})\oplus (N_{l-j})
# $$

# where the bitwise action of $\oplus$ means that if $N_{l-i}=N_{l-j}$ the result is
# $0$ whereas if $N_{l-i}\ne N_{l-j}$ the result is
# $1$. As an example, consider the case where  $N_{l-i}=6$ and $N_{l-j}=11$. The first
# one has a bit representation (using 4 bits only) which reads $0110$ whereas the 
# second number is $1011$. Employing the $\oplus$ operator yields 
# $1101$, or $2^3+2^2+2^0=13$.
# 
# In Fortran90, the bitwise $\oplus$ operation is coded through the intrinsic
# function $\mathrm{IEOR}(m,n)$ where $m$ and $n$ are the input numbers, while in $C$
# it is given by $m\wedge n$. 
# 
# We show here how the linear congruential algorithm can be implemented, namely

# $$
# N_i=(aN_{i-1}) \mathrm{MOD} (M).
# $$

# However, since $a$ and $N_{i-1}$ are integers and their multiplication 
# could become greater than the standard 32 bit integer, there is a trick via 
# Schrage's algorithm which approximates the multiplication
# of large integers through the factorization

# $$
# M=aq+r,
# $$

# where we have defined

# $$
# q=[M/a],
# $$

# and

# $$
# r = M\hspace{0.1cm}\mathrm{MOD} \hspace{0.1cm}a.
# $$

# where the brackets denote integer division. In the code below the numbers 
# $q$ and $r$ are chosen so that $r < q$.
# 
# To see how this works we note first that

# <!-- Equation labels as ordinary links -->
# <div id="eq:rntrick1"></div>
# 
# $$
# \begin{equation}
# (aN_{i-1}) \mathrm{MOD} (M)= (aN_{i-1}-[N_{i-1}/q]M)\mathrm{MOD} (M),
# \label{eq:rntrick1} \tag{14}
# \end{equation}
# $$

# since we can add or subtract any integer multiple of $M$ from $aN_{i-1}$.
# The last term $[N_{i-1}/q]M\mathrm{MOD}(M)$ is zero since the integer division 
# $[N_{i-1}/q]$ just yields a constant which is multiplied with $M$. 
# 
# We can now rewrite Eq. ([14](#eq:rntrick1)) as

# <!-- Equation labels as ordinary links -->
# <div id="eq:rntrick2"></div>
# 
# $$
# \begin{equation}
# (aN_{i-1}) \mathrm{MOD} (M)= (aN_{i-1}-[N_{i-1}/q](aq+r))\mathrm{MOD} (M),
# \label{eq:rntrick2} \tag{15}
# \end{equation}
# $$

# which results

# <!-- Equation labels as ordinary links -->
# <div id="eq:rntrick3"></div>
# 
# $$
# \begin{equation}
# (aN_{i-1}) \mathrm{MOD} (M)= \left(a(N_{i-1}-[N_{i-1}/q]q)-[N_{i-1}/q]r)\right)\mathrm{MOD} (M),
# \label{eq:rntrick3} \tag{16}
# \end{equation}
# $$

# yielding

# <!-- Equation labels as ordinary links -->
# <div id="eq:rntrick4"></div>
# 
# $$
# \begin{equation}
# (aN_{i-1}) \mathrm{MOD} (M)= \left(a(N_{i-1}\mathrm{MOD} (q)) -[N_{i-1}/q]r)\right)\mathrm{MOD} (M).
# \label{eq:rntrick4} \tag{17}
# \end{equation}
# $$

# The term $[N_{i-1}/q]r$ is always smaller or equal $N_{i-1}(r/q)$ and with $r < q$ we obtain always a 
# number smaller than $N_{i-1}$, which is smaller than $M$. 
# And since the number $N_{i-1}\mathrm{MOD} (q)$ is between zero and $q-1$ then
# $a(N_{i-1}\mathrm{MOD} (q))< aq$. Combined with our definition of $q=[M/a]$ ensures that 
# this term is also smaller than $M$ meaning that both terms fit into a
# 32-bit signed integer. None of these two terms can be negative, but their difference could.
# The algorithm below adds $M$ if their difference is negative.
# Note that the program uses the bitwise $\oplus$ operator to generate
# the starting point for each generation of a random number. The period
# of $ran0$ is $\sim 2.1\times 10^{9}$. A special feature of this
# algorithm is that is should never be called with the initial seed 
# set to $0$. 
# 
# As mentioned previously, the underlying PDF for the generation of
# random numbers is the uniform distribution, meaning that the 
# probability for finding a number $x$ in the interval [0,1] is $p(x)=1$.
# 
# A random number generator should produce numbers which are uniformly distributed
# in this interval. The table  shows the distribution of $N=10000$ random
# numbers generated by the functions in the program library.
# We note in this table that the number of points in the various
# intervals $0.0-0.1$, $0.1-0.2$ etc are fairly close to $1000$, with some minor
# deviations. 
# 
# Two additional measures are the standard deviation $\sigma$ and the mean
# $\mu=\langle x\rangle$.
# 
# For the uniform distribution, the mean value $\mu$ is then

# $$
# \mu=\langle x\rangle=\frac{1}{2}
# $$

# while the standard deviation is

# $$
# \sigma=\sqrt{\langle x^2\rangle-\mu^2}=\frac{1}{\sqrt{12}}=0.2886.
# $$

# The various random number generators produce results which agree rather well with
# these limiting values. 
# 
# <table class="dotable" border="1">
# <thead>
# <tr><th align="center">$x$-bin </th> <th align="center"> ran0 </th> <th align="center"> ran1 </th> <th align="center"> ran2 </th> <th align="center"> ran3 </th> </tr>
# </thead>
# <tbody>
# <tr><td align="center">   0.0-0.1     </td> <td align="right">   1013      </td> <td align="right">   991       </td> <td align="right">   938       </td> <td align="right">   1047      </td> </tr>
# <tr><td align="center">   0.1-0.2     </td> <td align="right">   1002      </td> <td align="right">   1009      </td> <td align="right">   1040      </td> <td align="right">   1030      </td> </tr>
# <tr><td align="center">   0.2-0.3     </td> <td align="right">   989       </td> <td align="right">   999       </td> <td align="right">   1030      </td> <td align="right">   993       </td> </tr>
# <tr><td align="center">   0.3-0.4     </td> <td align="right">   939       </td> <td align="right">   960       </td> <td align="right">   1023      </td> <td align="right">   937       </td> </tr>
# <tr><td align="center">   0.4-0.5     </td> <td align="right">   1038      </td> <td align="right">   1001      </td> <td align="right">   1002      </td> <td align="right">   992       </td> </tr>
# <tr><td align="center">   0.5-0.6     </td> <td align="right">   1037      </td> <td align="right">   1047      </td> <td align="right">   1009      </td> <td align="right">   1009      </td> </tr>
# <tr><td align="center">   0.6-0.7     </td> <td align="right">   1005      </td> <td align="right">   989       </td> <td align="right">   1003      </td> <td align="right">   989       </td> </tr>
# <tr><td align="center">   0.7-0.8     </td> <td align="right">   986       </td> <td align="right">   962       </td> <td align="right">   985       </td> <td align="right">   954       </td> </tr>
# <tr><td align="center">   0.8-0.9     </td> <td align="right">   1000      </td> <td align="right">   1027      </td> <td align="right">   1009      </td> <td align="right">   1023      </td> </tr>
# <tr><td align="center">   0.9-1.0     </td> <td align="right">   991       </td> <td align="right">   1015      </td> <td align="right">   961       </td> <td align="right">   1026      </td> </tr>
# <tr><td align="center">   $\mu$       </td> <td align="right">   0.4997    </td> <td align="right">   0.5018    </td> <td align="right">   0.4992    </td> <td align="right">   0.4990    </td> </tr>
# <tr><td align="center">   $\sigma$    </td> <td align="right">   0.2882    </td> <td align="right">   0.2892    </td> <td align="right">   0.2861    </td> <td align="right">   0.2915    </td> </tr>
# </tbody>
# </table>
# 
# The following simple Python code plots the distribution of the produced random numbers using the linear congruential RNG employed by Python. The trend displayed in the previous table is seen rather clearly.

# In[4]:


#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random

# initialize the rng with a seed
random.seed() 
counts = 10000
values = np.zeros(counts)   
for i in range (1, counts, 1):
    values[i] = random.random()

# the histogram of the data
n, bins, patches = plt.hist(values, 10, facecolor='green')

plt.xlabel('$x$')
plt.ylabel('Number of counts')
plt.title(r'Test of uniform distribution')
plt.axis([0, 1, 0, 1100])
plt.grid(True)
plt.show()


# Since our random numbers, which are typically generated via a linear congruential algorithm,
# are never fully independent, we can then define 
# an important test which measures the degree of correlation, namely the  so-called  
# auto-correlation function defined previously, see again Eq. ([9](#eq:autocorrelformal)).
# We rewrite it here as

# $$
# C_k=\frac{f_d}
#              {\sigma^2},
# $$

# with $C_0=1$. Recall that 
# $\sigma^2=\langle x_i^2\rangle-\langle x_i\rangle^2$ and that

# $$
# f_d = \frac{1}{nm}\sum_{\alpha=1}^m\sum_{k=1}^{n-d}(x_{\alpha,k}-\langle X_m \rangle)(x_{\alpha,k+d}-\langle X_m \rangle),
# $$

# The non-vanishing of $C_k$ for $k\ne 0$ means that the random
# numbers are not independent. The independence of the random numbers is crucial 
# in the evaluation of other expectation values. If they are not independent, our
# assumption for approximating $\sigma_N$ is no longer valid.

# ### Autocorrelation function
# 
# This program computes the autocorrelation function as discussed in the equation on the previous slide for random numbers generated with the normal distribution $N(0,1)$.

# In[5]:


# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

def autocovariance(x, n, k, mean_x):
    sum = 0.0
    for i in range(0, n-k):
        sum += (x[(i+k)]-mean_x)*(x[i]-mean_x)
    return  sum/n

n = 1000
x=np.random.normal(size=n)
autocor = np.zeros(n)
figaxis = np.zeros(n)
mean_x=np.mean(x)
var_x = np.var(x)
print(mean_x, var_x)
for i in range (0, n):
    figaxis[i] = i
    autocor[i]=(autocovariance(x, n, i, mean_x))/var_x    

plt.plot(figaxis, autocor, "r-")
plt.axis([0,n,-0.1, 1.0])
plt.xlabel(r'$i$')
plt.ylabel(r'$\gamma_i$')
plt.title(r'Autocorrelation function')
plt.show()


# As can be seen from the plot, the first point gives back the variance and a value of one. 
# For the remaining values we notice that there are still non-zero values for the auto-correlation function.
