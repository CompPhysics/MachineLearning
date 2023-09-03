#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html week36.do.txt --no_mako -->
# <!-- dom:TITLE: Week 36: Statistical interpretation of Linear Regression and Resampling techniques -->

# # Week 36: Statistical interpretation of Linear Regression and Resampling techniques
# **Morten Hjorth-Jensen**, Department of Physics, University of Oslo and Department of Physics and Astronomy and National Superconducting Cyclotron Laboratory, Michigan State University
# 
# Date: **September 4-8, 2023**

# ## Plans for week 36
# 
# * Material for the active learning sessions on Tuesday and Wednesday
# 
#   * Summary from last week on discussion of SVD, Ridge and Lasso linear regression.
# 
#   * Recommended Reading: Hastie et al chapter 3, see <https://link.springer.com/book/10.1007/978-0-387-84858-7>
# 
#   * Presentation and discussion of first project
# 
# * Material for the lecture on Thursday September 7
# 
#   * Linear Regression and links with Statistics, Resampling methods
# 
#   * Recommended Reading: Goodfellow et al chapter 3 on probability theory, see URL:""
# 
#   * See also Murphy, sections 2.4 (Gaussian distributions) and 3.2 (Bayesian Statistics, basis)

# ## Material for the active learning sessions Tuesday and Wednesday
# 
# The material here contains a summary from last Week and discussion of SVD, Ridge and Lasso regression with examples

# ## Linear Regression and  the SVD
# 
# We used the SVD to analyse the matrix to invert in ordinary lineat regression

# $$
# \boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U}^T\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T=\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\boldsymbol{V}^T.
# $$

# Since the matrices here have dimension $p\times p$, with $p$ corresponding to the singular values, we defined last week the matrix

# $$
# \boldsymbol{\Sigma}^T\boldsymbol{\Sigma} = \begin{bmatrix} \tilde{\boldsymbol{\Sigma}} & \boldsymbol{0}\\ \end{bmatrix}\begin{bmatrix} \tilde{\boldsymbol{\Sigma}} \\ \boldsymbol{0}\end{bmatrix},
# $$

# where the tilde-matrix $\tilde{\boldsymbol{\Sigma}}$ is a matrix of dimension $p\times p$ containing only the singular values $\sigma_i$, that is

# $$
# \tilde{\boldsymbol{\Sigma}}=\begin{bmatrix} \sigma_0 & 0 & 0 & \dots & 0 & 0 \\
#                                     0 & \sigma_1 & 0 & \dots & 0 & 0 \\
# 				    0 & 0 & \sigma_2 & \dots & 0 & 0 \\
# 				    0 & 0 & 0 & \dots & \sigma_{p-2} & 0 \\
# 				    0 & 0 & 0 & \dots & 0 & \sigma_{p-1} \\
# \end{bmatrix},
# $$

# meaning we can write

# $$
# \boldsymbol{X}^T\boldsymbol{X}=\boldsymbol{V}\tilde{\boldsymbol{\Sigma}}^2\boldsymbol{V}^T.
# $$

# Multiplying from the right with $\boldsymbol{V}$ (using the orthogonality of $\boldsymbol{V}$) we get

# $$
# \left(\boldsymbol{X}^T\boldsymbol{X}\right)\boldsymbol{V}=\boldsymbol{V}\tilde{\boldsymbol{\Sigma}}^2.
# $$

# ## What does it mean?
# 
# This means the vectors $\boldsymbol{v}_i$ of the orthogonal matrix $\boldsymbol{V}$
# are the eigenvectors of the matrix $\boldsymbol{X}^T\boldsymbol{X}$ with eigenvalues
# given by the singular values squared, that is

# $$
# \left(\boldsymbol{X}^T\boldsymbol{X}\right)\boldsymbol{v}_i=\boldsymbol{v}_i\sigma_i^2.
# $$

# In other words, each non-zero singular value of $\boldsymbol{X}$ is a positive
# square root of an eigenvalue of $\boldsymbol{X}^T\boldsymbol{X}$.  It means also that
# the columns of $\boldsymbol{V}$ are the eigenvectors of
# $\boldsymbol{X}^T\boldsymbol{X}$. Since we have ordered the singular values of
# $\boldsymbol{X}$ in a descending order, it means that the column vectors
# $\boldsymbol{v}_i$ are hierarchically ordered by how much correlation they
# encode from the columns of $\boldsymbol{X}$. 
# 
# Note that these are also the eigenvectors and eigenvalues of the
# Hessian matrix.
# 
# If we now recall the definition of the covariance matrix (not using
# Bessel's correction) we have

# $$
# \boldsymbol{C}[\boldsymbol{X}]=\frac{1}{n}\boldsymbol{X}^T\boldsymbol{X},
# $$

# meaning that every squared non-singular value of $\boldsymbol{X}$ divided by $n$ (
# the number of samples) are the eigenvalues of the covariance
# matrix. Every singular value of $\boldsymbol{X}$ is thus a positive square
# root of an eigenvalue of $\boldsymbol{X}^T\boldsymbol{X}$. If the matrix $\boldsymbol{X}$ is
# self-adjoint, the singular values of $\boldsymbol{X}$ are equal to the
# absolute value of the eigenvalues of $\boldsymbol{X}$.

# ## And finally  $\boldsymbol{X}\boldsymbol{X}^T$
# 
# For $\boldsymbol{X}\boldsymbol{X}^T$ we found

# $$
# \boldsymbol{X}\boldsymbol{X}^T=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T\boldsymbol{V}\boldsymbol{\Sigma}^T\boldsymbol{U}^T=\boldsymbol{U}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\boldsymbol{U}^T.
# $$

# Since the matrices here have dimension $n\times n$, we have

# $$
# \boldsymbol{\Sigma}\boldsymbol{\Sigma}^T = \begin{bmatrix} \tilde{\boldsymbol{\Sigma}} \\ \boldsymbol{0}\\ \end{bmatrix}\begin{bmatrix} \tilde{\boldsymbol{\Sigma}}  \boldsymbol{0}\\ \end{bmatrix}=\begin{bmatrix} \tilde{\boldsymbol{\Sigma}} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0}\\ \end{bmatrix},
# $$

# leading to

# $$
# \boldsymbol{X}\boldsymbol{X}^T=\boldsymbol{U}\begin{bmatrix} \tilde{\boldsymbol{\Sigma}} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0}\\ \end{bmatrix}\boldsymbol{U}^T.
# $$

# Multiplying with $\boldsymbol{U}$ from the right gives us the eigenvalue problem

# $$
# (\boldsymbol{X}\boldsymbol{X}^T)\boldsymbol{U}=\boldsymbol{U}\begin{bmatrix} \tilde{\boldsymbol{\Sigma}} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0}\\ \end{bmatrix}.
# $$

# It means that the eigenvalues of $\boldsymbol{X}\boldsymbol{X}^T$ are again given by
# the non-zero singular values plus now a series of zeros.  The column
# vectors of $\boldsymbol{U}$ are the eigenvectors of $\boldsymbol{X}\boldsymbol{X}^T$ and
# measure how much correlations are contained in the rows of $\boldsymbol{X}$.
# 
# Since we will mainly be interested in the correlations among the features
# of our data (the columns of $\boldsymbol{X}$, the quantity of interest for us are the non-zero singular
# values and the column vectors of $\boldsymbol{V}$.

# ## Code for SVD and Inversion of Matrices
# 
# How do we use the SVD to invert a matrix $\boldsymbol{X}^\boldsymbol{X}$ which is singular or near singular?
# The simple answer is to use the linear algebra function for pseudoinvers, that is

# In[1]:


Ainv = np.linlag.pinv(A)


# Let us first look at a matrix which does not causes problems and write our own function where we just use the SVD.

# In[2]:


import numpy as np
# SVD inversion
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    print('test U')
    print( (np.transpose(U) @ U - U @np.transpose(U)))
    print('test VT')
    print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))


    D = np.zeros((len(U),len(VT)))
    D = np.diag(s)
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))


#X = np.array([ [1.0, -1.0, 2.0], [1.0, 0.0, 1.0], [1.0, 2.0, -1.0], [1.0, 1.0, 0.0] ])
# Non-singular square matrix
X = np.array( [ [1,2,3],[2,4,5],[3,5,6]])
print(X)
A = np.transpose(X) @ X
# Brute force inversion
B = np.linalg.inv(A)  # here we could use np.linalg.pinv(A)
C = SVDinv(A)
print(np.abs(B-C))


# ## Inverse of Rectangular Matrix
# 
# Although our matrix to invert $\boldsymbol{X}^T\boldsymbol{X}$ is a square matrix, our matrix may be singular. 
# 
# The pseudoinverse is the generalization of the matrix inverse for square matrices to
# rectangular matrices where the number of rows and columns are not equal.
# 
# It is also called the the Moore-Penrose Inverse after two independent discoverers of the method or the Generalized Inverse.
# It is used for the calculation of the inverse for singular or near singular matrices and for rectangular matrices.
# 
# Using the SVD we can obtain the pseudoinverse of a matrix $\boldsymbol{A}$ (labeled here as $\boldsymbol{A}_{\mathrm{PI}}$)

# $$
# \boldsymbol{A}_{\mathrm{PI}}= \boldsymbol{V}\boldsymbol{D}_{\mathrm{PI}}\boldsymbol{U}^T,
# $$

# where $\boldsymbol{D}_{\mathrm{PI}}$ can be calculated by creating a diagonal matrix from $\boldsymbol{\Sigma}$ where we only keep the singular values (the non-zero values). The following code computes the pseudoinvers of the matrix based on the SVD.

# In[3]:


import numpy as np
# SVD inversion
def SVDinv(A):
    U, s, VT = np.linalg.svd(A)
    # reciprocals of singular values of s
    d = 1.0 / s
    # create m x n D matrix
    D = np.zeros(A.shape)
    # populate D with n x n diagonal matrix
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    UT = np.transpose(U)
    V = np.transpose(VT)
    return np.matmul(V,np.matmul(D.T,UT))


A = np.array([ [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],[0.9, 1.0]])
print(A)
# Brute force inversion of super-collinear matrix
B = np.linalg.pinv(A)
print(B)
# Compare our own algorithm with pinv
C = SVDinv(A)
print(np.abs(C-B))


# As you can see from this example, our own decomposition based on the SVD agrees with  the pseudoinverse algorithm provided by **Numpy**.

# ## Ridge and LASSO Regression
# 
# Let us remind ourselves about the expression for the standard Mean Squared Error (MSE) which we used to define our cost function and the equations for the ordinary least squares (OLS) method, that is 
# our optimization problem is

# $$
# {\displaystyle \min_{\boldsymbol{\beta}\in {\mathbb{R}}^{p}}}\frac{1}{n}\left\{\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right)\right\}.
# $$

# or we can state it as

# $$
# {\displaystyle \min_{\boldsymbol{\beta}\in
# {\mathbb{R}}^{p}}}\frac{1}{n}\sum_{i=0}^{n-1}\left(y_i-\tilde{y}_i\right)^2=\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\vert\vert_2^2,
# $$

# where we have used the definition of  a norm-2 vector, that is

# $$
# \vert\vert \boldsymbol{x}\vert\vert_2 = \sqrt{\sum_i x_i^2}.
# $$

# ## From OLS to Ridge and Lasso
# 
# By minimizing the above equation with respect to the parameters
# $\boldsymbol{\beta}$ we could then obtain an analytical expression for the
# parameters $\boldsymbol{\beta}$.  We can add a regularization parameter $\lambda$ by
# defining a new cost function to be optimized, that is

# $$
# {\displaystyle \min_{\boldsymbol{\beta}\in
# {\mathbb{R}}^{p}}}\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\vert\vert_2^2+\lambda\vert\vert \boldsymbol{\beta}\vert\vert_2^2
# $$

# which leads to the Ridge regression minimization problem where we
# require that $\vert\vert \boldsymbol{\beta}\vert\vert_2^2\le t$, where $t$ is
# a finite number larger than zero. We do not include such a constraints in the discussions here.
# 
# By defining

# $$
# C(\boldsymbol{X},\boldsymbol{\beta})=\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\vert\vert_2^2+\lambda\vert\vert \boldsymbol{\beta}\vert\vert_1,
# $$

# we have a new optimization equation

# $$
# {\displaystyle \min_{\boldsymbol{\beta}\in
# {\mathbb{R}}^{p}}}\frac{1}{n}\vert\vert \boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\vert\vert_2^2+\lambda\vert\vert \boldsymbol{\beta}\vert\vert_1
# $$

# which leads to Lasso regression. Lasso stands for least absolute shrinkage and selection operator. 
# 
# Here we have defined the norm-1 as

# $$
# \vert\vert \boldsymbol{x}\vert\vert_1 = \sum_i \vert x_i\vert.
# $$

# ## Deriving the  Ridge Regression Equations
# 
# Using the matrix-vector expression for Ridge regression and dropping the parameter $1/n$ in front of the standard means squared error equation, we have

# $$
# C(\boldsymbol{X},\boldsymbol{\beta})=\left\{(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\right\}+\lambda\boldsymbol{\beta}^T\boldsymbol{\beta},
# $$

# and 
# taking the derivatives with respect to $\boldsymbol{\beta}$ we obtain then
# a slightly modified matrix inversion problem which for finite values
# of $\lambda$ does not suffer from singularity problems. We obtain
# the optimal parameters

# $$
# \hat{\boldsymbol{\beta}}_{\mathrm{Ridge}} = \left(\boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y},
# $$

# with $\boldsymbol{I}$ being a $p\times p$ identity matrix with the constraint that

# $$
# \sum_{i=0}^{p-1} \beta_i^2 \leq t,
# $$

# with $t$ a finite positive number.

# ## Note on Scikit-Learn
# 
# Note well that a library like **Scikit-Learn** does not include the $1/n$ factor in the expression for the mean-squared error. If you include it, the optimal parameter $\beta$ becomes

# $$
# \hat{\boldsymbol{\beta}}_{\mathrm{Ridge}} = \left(\boldsymbol{X}^T\boldsymbol{X}+n\lambda\boldsymbol{I}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}.
# $$

# In our codes where we compare our own codes with **Scikit-Learn**, we do thus not include the $1/n$ factor in the cost function.

# ## Comparison with OLS
# When we compare this with the ordinary least squares result we have

# $$
# \hat{\boldsymbol{\beta}}_{\mathrm{OLS}} = \left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y},
# $$

# which can lead to singular matrices. However, with the SVD, we can always compute the inverse of the matrix $\boldsymbol{X}^T\boldsymbol{X}$.
# 
# We see that Ridge regression is nothing but the standard OLS with a
# modified diagonal term added to $\boldsymbol{X}^T\boldsymbol{X}$. The consequences, in
# particular for our discussion of the bias-variance tradeoff are rather
# interesting. We will see that for specific values of $\lambda$, we may
# even reduce the variance of the optimal parameters $\boldsymbol{\beta}$. These topics and other related ones, will be discussed after the more linear algebra oriented analysis here.

# ## SVD analysis
# 
# Using our insights about the SVD of the design matrix $\boldsymbol{X}$ 
# We have already analyzed the OLS solutions in terms of the eigenvectors (the columns) of the right singular value matrix $\boldsymbol{U}$ as

# $$
# \tilde{\boldsymbol{y}}_{\mathrm{OLS}}=\boldsymbol{X}\boldsymbol{\beta}  =\boldsymbol{U}\boldsymbol{U}^T\boldsymbol{y}.
# $$

# For Ridge regression this becomes

# $$
# \tilde{\boldsymbol{y}}_{\mathrm{Ridge}}=\boldsymbol{X}\boldsymbol{\beta}_{\mathrm{Ridge}} = \boldsymbol{U\Sigma V^T}\left(\boldsymbol{V}\boldsymbol{\Sigma}^2\boldsymbol{V}^T+\lambda\boldsymbol{I} \right)^{-1}(\boldsymbol{U\Sigma V^T})^T\boldsymbol{y}=\sum_{j=0}^{p-1}\boldsymbol{u}_j\boldsymbol{u}_j^T\frac{\sigma_j^2}{\sigma_j^2+\lambda}\boldsymbol{y},
# $$

# with the vectors $\boldsymbol{u}_j$ being the columns of $\boldsymbol{U}$ from the SVD of the matrix $\boldsymbol{X}$.

# ## Interpreting the Ridge results
# 
# Since $\lambda \geq 0$, it means that compared to OLS, we have

# $$
# \frac{\sigma_j^2}{\sigma_j^2+\lambda} \leq 1.
# $$

# Ridge regression finds the coordinates of $\boldsymbol{y}$ with respect to the
# orthonormal basis $\boldsymbol{U}$, it then shrinks the coordinates by
# $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$. Recall that the SVD has
# eigenvalues ordered in a descending way, that is $\sigma_i \geq
# \sigma_{i+1}$.
# 
# For small eigenvalues $\sigma_i$ it means that their contributions become less important, a fact which can be used to reduce the number of degrees of freedom. More about this when we have covered the material on a statistical interpretation of various linear regression methods.

# ## More interpretations
# 
# For the sake of simplicity, let us assume that the design matrix is orthonormal, that is

# $$
# \boldsymbol{X}^T\boldsymbol{X}=(\boldsymbol{X}^T\boldsymbol{X})^{-1} =\boldsymbol{I}.
# $$

# In this case the standard OLS results in

# $$
# \boldsymbol{\beta}^{\mathrm{OLS}} = \boldsymbol{X}^T\boldsymbol{y}=\sum_{i=0}^{n-1}\boldsymbol{u}_i\boldsymbol{u}_i^T\boldsymbol{y},
# $$

# and

# $$
# \boldsymbol{\beta}^{\mathrm{Ridge}} = \left(\boldsymbol{I}+\lambda\boldsymbol{I}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}=\left(1+\lambda\right)^{-1}\boldsymbol{\beta}^{\mathrm{OLS}},
# $$

# that is the Ridge estimator scales the OLS estimator by the inverse of a factor $1+\lambda$, and
# the Ridge estimator converges to zero when the hyperparameter goes to
# infinity.
# 
# We will come back to more interpreations after we have gone through some of the statistical analysis part. 
# 
# For more discussions of Ridge and Lasso regression, [Wessel van Wieringen's](https://arxiv.org/abs/1509.09169) article is highly recommended.
# Similarly, [Mehta et al's article](https://arxiv.org/abs/1803.08823) is also recommended.

# ## Deriving the  Lasso Regression Equations
# 
# Using the matrix-vector expression for Lasso regression, we have the following **cost** function

# $$
# C(\boldsymbol{X},\boldsymbol{\beta})=\frac{1}{n}\left\{(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\right\}+\lambda\vert\vert\boldsymbol{\beta}\vert\vert_1,
# $$

# Taking the derivative with respect to $\boldsymbol{\beta}$ and recalling that the derivative of the absolute value is (we drop the boldfaced vector symbol for simplicity)

# $$
# \frac{d \vert \beta\vert}{d \beta}=\mathrm{sgn}(\beta)=\left\{\begin{array}{cc} 1 & \beta > 0 \\-1 & \beta < 0, \end{array}\right.
# $$

# we have that the derivative of the cost function is

# $$
# \frac{\partial C(\boldsymbol{X},\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}=-\frac{2}{n}\boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})+\lambda sgn(\boldsymbol{\beta})=0,
# $$

# and reordering we have

# $$
# \boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\beta}+\lambda sgn(\boldsymbol{\beta})=\boldsymbol{X}^T\boldsymbol{y}.
# $$

# This equation does not lead to a nice analytical equation as in Ridge regression or ordinary least squares. We have absorbed the factor $2/n$ in a redefinition of the parameter $\lambda$. We will solve this type of problems using libraries like **scikit-learn**.

# ## Simple example to illustrate Ordinary Least Squares, Ridge and Lasso Regression
# 
# Let us assume that our design matrix is given by unit (identity) matrix, that is a square diagonal matrix with ones only along the
# diagonal. In this case we have an equal number of rows and columns $n=p$.
# 
# Our model approximation is just $\tilde{\boldsymbol{y}}=\boldsymbol{\beta}$ and the mean squared error and thereby the cost function for ordinary least sqquares (OLS) is then (we drop the term $1/n$)

# $$
# C(\boldsymbol{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2,
# $$

# and minimizing we have that

# $$
# \hat{\beta}_i^{\mathrm{OLS}} = y_i.
# $$

# ## Ridge Regression
# 
# For Ridge regression our cost function is

# $$
# C(\boldsymbol{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\beta_i^2,
# $$

# and minimizing we have that

# $$
# \hat{\beta}_i^{\mathrm{Ridge}} = \frac{y_i}{1+\lambda}.
# $$

# ## Lasso Regression
# 
# For Lasso regression our cost function is

# $$
# C(\boldsymbol{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\vert\beta_i\vert=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\sqrt{\beta_i^2},
# $$

# and minimizing we have that

# $$
# -2\sum_{i=0}^{p-1}(y_i-\beta_i)+\lambda \sum_{i=0}^{p-1}\frac{(\beta_i)}{\vert\beta_i\vert}=0,
# $$

# which leads to

# $$
# \hat{\boldsymbol{\beta}}_i^{\mathrm{Lasso}} = \left\{\begin{array}{ccc}y_i-\frac{\lambda}{2} &\mathrm{if} & y_i> \frac{\lambda}{2}\\
#                                                           y_i+\frac{\lambda}{2} &\mathrm{if} & y_i< -\frac{\lambda}{2}\\
# 							  0 &\mathrm{if} & \vert y_i\vert\le  \frac{\lambda}{2}\end{array}\right.\\.
# $$

# Plotting these results shows clearly that Lasso regression suppresses (sets to zero) values of $\beta_i$ for specific values of $\lambda$. Ridge regression reduces on the other hand the values of $\beta_i$ as function of $\lambda$.

# ## Yet another Example
# 
# Let us assume we have a data set with outputs/targets given by the vector

# $$
# \boldsymbol{y}=\begin{bmatrix}4 \\ 2 \\3\end{bmatrix},
# $$

# and our inputs as a $3\times 2$ design matrix

# $$
# \boldsymbol{X}=\begin{bmatrix}2 & 0\\ 0 & 1 \\ 0 & 0\end{bmatrix},
# $$

# meaning that we have two features and two unknown parameters $\beta_0$ and $\beta_1$ to be determined either by ordinary least squares, Ridge or Lasso regression.

# ## The OLS case
# 
# For ordinary least squares (OLS) we know that the optimal solution is

# $$
# \hat{\boldsymbol{\beta}}^{\mathrm{OLS}}=\left( \boldsymbol{X}^T\boldsymbol{X}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}.
# $$

# Inserting the above values we obtain that

# $$
# \hat{\boldsymbol{\beta}}^{\mathrm{OLS}}=\begin{bmatrix}2 \\ 2\end{bmatrix},
# $$

# The code which implements this simpler case is presented after the discussion of Ridge and Lasso.

# ## The Ridge case
# 
# For Ridge regression we have

# $$
# \hat{\boldsymbol{\beta}}^{\mathrm{Ridge}}=\left( \boldsymbol{X}^T\boldsymbol{X}+\lambda\boldsymbol{I}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}.
# $$

# Inserting the above values we obtain that

# $$
# \hat{\boldsymbol{\beta}}^{\mathrm{Ridge}}=\begin{bmatrix}\frac{8}{4+\lambda} \\ \frac{2}{1+\lambda}\end{bmatrix},
# $$

# There is normally a constraint on the value of $\vert\vert \boldsymbol{\beta}\vert\vert_2$ via the parameter $\lambda$.
# Let us for simplicity assume that $\beta_0^2+\beta_1^2=1$ as constraint. This will allow us to find an expression for the optimal values of $\beta$ and $\lambda$.
# 
# To see this, let us write the cost function for Ridge regression.

# ## Writing the Cost Function
# 
# We define the MSE without the $1/n$ factor and have then, using that

# $$
# \boldsymbol{X}\boldsymbol{\beta}=\begin{bmatrix} 2\beta_0 \\ \beta_1 \\0 \end{bmatrix},
# $$

# $$
# C(\boldsymbol{\beta})=(4-2\beta_0)^2+(2-\beta_1)^2+\lambda(\beta_0^2+\beta_1^2),
# $$

# and taking the derivative with respect to $\beta_0$ we get

# $$
# \beta_0=\frac{8}{4+\lambda},
# $$

# and for $\beta_1$ we obtain

# $$
# \beta_1=\frac{2}{1+\lambda},
# $$

# Using the constraint for $\beta_0^2+\beta_1^2=1$ we can constrain $\lambda$ by solving

# $$
# \left(\frac{8}{4+\lambda}\right)^2+\left(\frac{2}{1+\lambda}\right)^2=1,
# $$

# which gives $\lambda=4.571$ and $\beta_0=0.933$ and $\beta_1=0.359$.

# ## Lasso case
# 
# For Lasso we need now, keeping a  constraint on $\vert\beta_0\vert+\vert\beta_1\vert=1$,  to take the derivative of the absolute values of $\beta_0$
# and $\beta_1$. This gives us the following derivatives of the cost function

# $$
# C(\boldsymbol{\beta})=(4-2\beta_0)^2+(2-\beta_1)^2+\lambda(\vert\beta_0\vert+\vert\beta_1\vert),
# $$

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \beta_0}=-4(4-2\beta_0)+\lambda\mathrm{sgn}(\beta_0)=0,
# $$

# and

# $$
# \frac{\partial C(\boldsymbol{\beta})}{\partial \beta_1}=-2(2-\beta_1)+\lambda\mathrm{sgn}(\beta_1)=0.
# $$

# We have now four cases to solve besides the trivial cases $\beta_0$ and/or $\beta_1$ are zero, namely
# 1. $\beta_0 > 0$ and $\beta_1 > 0$,
# 
# 2. $\beta_0 > 0$ and $\beta_1 < 0$,
# 
# 3. $\beta_0 < 0$ and $\beta_1 > 0$,
# 
# 4. $\beta_0 < 0$ and $\beta_1 < 0$.

# ## The first Case
# 
# If we consider the first case, we have then

# $$
# -4(4-2\beta_0)+\lambda=0,
# $$

# and

# $$
# -2(2-\beta_1)+\lambda=0.
# $$

# which yields

# $$
# \beta_0=\frac{16+\lambda}{8},
# $$

# and

# $$
# \beta_1=\frac{4+\lambda}{2}.
# $$

# Using the constraint on $\beta_0$ and $\beta_1$ we can then find the optimal value of $\lambda$ for the different cases. We leave this as an exercise to you.

# ## Simple code for solving the above problem
# 
# Here we set up the OLS, Ridge and Lasso functionality in order to study the above example. Note that here we have opted for a set of values of $\lambda$, meaning that we need to perform a search in order to find the optimal values.
# 
# First we study and compare the OLS and Ridge results.  The next code compares all three methods.

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.

X = np.array( [ [ 2, 0], [0, 1], [0,0]])
y = np.array( [4, 2, 3])


# matrix inversion to find beta
OLSbeta = np.linalg.inv(X.T @ X) @ X.T @ y
print(OLSbeta)
# and then make the prediction
ytildeOLS = X @ OLSbeta
print("Training MSE for OLS")
print(MSE(y,ytildeOLS))
ypredictOLS = X @ OLSbeta

# Repeat now for Ridge regression and various values of the regularization parameter
I = np.eye(2,2)
# Decide which values of lambda to use
nlambdas = 100
MSEPredict = np.zeros(nlambdas)
lambdas = np.logspace(-4, 4, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.inv(X.T @ X+lmb*I) @ X.T @ y
#    print(Ridgebeta)
    # and then make the prediction
    ypredictRidge = X @ Ridgebeta
    MSEPredict[i] = MSE(y,ypredictRidge)
#    print(MSEPredict[i])
    # Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Train')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# We see here that we reach a plateau. What is actually happening?

# ## With Lasso Regression

# In[5]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.

X = np.array( [ [ 2, 0], [0, 1], [0,0]])
y = np.array( [4, 2, 3])


# matrix inversion to find beta
OLSbeta = np.linalg.inv(X.T @ X) @ X.T @ y
print(OLSbeta)
# and then make the prediction
ytildeOLS = X @ OLSbeta
print("Training MSE for OLS")
print(MSE(y,ytildeOLS))
ypredictOLS = X @ OLSbeta

# Repeat now for Ridge regression and various values of the regularization parameter
I = np.eye(2,2)
# Decide which values of lambda to use
nlambdas = 100
MSERidgePredict = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
lambdas = np.logspace(-4, 4, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.inv(X.T @ X+lmb*I) @ X.T @ y
    print(Ridgebeta)
    # and then make the prediction
    ypredictRidge = X @ Ridgebeta
    MSERidgePredict[i] = MSE(y,ypredictRidge)
    RegLasso = linear_model.Lasso(lmb,fit_intercept=False)
    RegLasso.fit(X,y)
    ypredictLasso = RegLasso.predict(X)
    print(RegLasso.coef_)
    MSELassoPredict[i] = MSE(y,ypredictLasso)
# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSERidgePredict, 'r--', label = 'MSE Ridge Train')
plt.plot(np.log10(lambdas), MSELassoPredict, 'r--', label = 'MSE Lasso Train')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ## Another Example, now with a polynomial fit

# In[6]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
print("Training MSE for OLS")
print(MSE(y_train,ytildeOLS))
ypredictOLS = X_test @ OLSbeta
print("Test MSE OLS")
print(MSE(y_test,ypredictOLS))

# Repeat now for Lasso and Ridge regression and various values of the regularization parameter
I = np.eye(p,p)
# Decide which values of lambda to use
nlambdas = 100
MSEPredict = np.zeros(nlambdas)
MSETrain = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
MSELassoTrain = np.zeros(nlambdas)
lambdas = np.logspace(-4, 4, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    Ridgebeta = np.linalg.inv(X_train.T @ X_train+lmb*I) @ X_train.T @ y_train
    # include lasso using Scikit-Learn
    RegLasso = linear_model.Lasso(lmb,fit_intercept=False)
    RegLasso.fit(X_train,y_train)
    # and then make the prediction
    ytildeRidge = X_train @ Ridgebeta
    ypredictRidge = X_test @ Ridgebeta
    ytildeLasso = RegLasso.predict(X_train)
    ypredictLasso = RegLasso.predict(X_test)
    MSEPredict[i] = MSE(y_test,ypredictRidge)
    MSETrain[i] = MSE(y_train,ytildeRidge)
    MSELassoPredict[i] = MSE(y_test,ypredictLasso)
    MSELassoTrain[i] = MSE(y_train,ytildeLasso)

# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSETrain, label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSELassoTrain, label = 'MSE Lasso train')
plt.plot(np.log10(lambdas), MSELassoPredict, 'r--', label = 'MSE Lasso Test')

plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ## Material for lecture Thursday September 7

# ## Linking the regression analysis with a statistical interpretation
# 
# We will now couple the discussions of ordinary least squares, Ridge
# and Lasso regression with a statistical interpretation, that is we
# move from a linear algebra analysis to a statistical analysis. In
# particular, we will focus on what the regularization terms can result
# in.  We will amongst other things show that the regularization
# parameter can reduce considerably the variance of the parameters
# $\beta$.
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

# ## Assumptions made
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

# ## Expectation value and variance
# 
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

# ## Expectation value and variance for $\boldsymbol{\beta}$
# 
# With the OLS expressions for the optimal parameters $\boldsymbol{\hat{\beta}}$ we can evaluate the expectation value

# $$
# \mathbb{E}(\boldsymbol{\hat{\beta}}) = \mathbb{E}[ (\mathbf{X}^{\top} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbb{E}[ \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T}\mathbf{X}\boldsymbol{\beta}=\boldsymbol{\beta}.
# $$

# This means that the estimator of the regression parameters is unbiased.
# 
# We can also calculate the variance
# 
# The variance of the optimal value $\boldsymbol{\hat{\beta}}$ is

# $$
# \begin{eqnarray*}
# \mbox{Var}(\boldsymbol{\hat{\beta}}) & = & \mathbb{E} \{ [\boldsymbol{\beta} - \mathbb{E}(\boldsymbol{\beta})] [\boldsymbol{\beta} - \mathbb{E}(\boldsymbol{\beta})]^{T} \}
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
# $\boldsymbol{\sigma}^2 (\boldsymbol{\beta}_j ) = \boldsymbol{\sigma}^2 [(\mathbf{X}^{T} \mathbf{X})^{-1}]_{jj} $. This may be used to
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

# ## Deriving OLS from a probability distribution
# 
# Our basic assumption when we derived the OLS equations was to assume
# that our output is determined by a given continuous function
# $f(\boldsymbol{x})$ and a random noise $\boldsymbol{\epsilon}$ given by the normal
# distribution with zero mean value and an undetermined variance
# $\sigma^2$.
# 
# We found above that the outputs $\boldsymbol{y}$ have a mean value given by
# $\boldsymbol{X}\hat{\boldsymbol{\beta}}$ and variance $\sigma^2$. Since the entries to
# the design matrix are not stochastic variables, we can assume that the
# probability distribution of our targets is also a normal distribution
# but now with mean value $\boldsymbol{X}\hat{\boldsymbol{\beta}}$. This means that a
# single output $y_i$ is given by the Gaussian distribution

# $$
# y_i\sim \mathcal{N}(\boldsymbol{X}_{i,*}\boldsymbol{\beta}, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}.
# $$

# ## Independent and Identically Distrubuted (iid)
# 
# We assume now that the various $y_i$ values are stochastically distributed according to the above Gaussian distribution. 
# We define this distribution as

# $$
# p(y_i, \boldsymbol{X}\vert\boldsymbol{\beta})=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]},
# $$

# which reads as finding the likelihood of an event $y_i$ with the input variables $\boldsymbol{X}$ given the parameters (to be determined) $\boldsymbol{\beta}$.
# 
# Since these events are assumed to be independent and identicall distributed we can build the probability distribution function (PDF) for all possible event $\boldsymbol{y}$ as the product of the single events, that is we have

# $$
# p(\boldsymbol{y},\boldsymbol{X}\vert\boldsymbol{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}=\prod_{i=0}^{n-1}p(y_i,\boldsymbol{X}\vert\boldsymbol{\beta}).
# $$

# We will write this in a more compact form reserving $\boldsymbol{D}$ for the domain of events, including the ouputs (targets) and the inputs. That is
# in case we have a simple one-dimensional input and output case

# $$
# \boldsymbol{D}=[(x_0,y_0), (x_1,y_1),\dots, (x_{n-1},y_{n-1})].
# $$

# In the more general case the various inputs should be replaced by the possible features represented by the input data set $\boldsymbol{X}$. 
# We can now rewrite the above probability as

# $$
# p(\boldsymbol{D}\vert\boldsymbol{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}.
# $$

# It is a conditional probability (see below) and reads as the likelihood of a domain of events $\boldsymbol{D}$ given a set of parameters $\boldsymbol{\beta}$.

# ## Maximum Likelihood Estimation (MLE)
# 
# In statistics, maximum likelihood estimation (MLE) is a method of
# estimating the parameters of an assumed probability distribution,
# given some observed data. This is achieved by maximizing a likelihood
# function so that, under the assumed statistical model, the observed
# data is the most probable. 
# 
# We will assume here that our events are given by the above Gaussian
# distribution and we will determine the optimal parameters $\beta$ by
# maximizing the above PDF. However, computing the derivatives of a
# product function is cumbersome and can easily lead to overflow and/or
# underflowproblems, with potentials for loss of numerical precision.
# 
# In practice, it is more convenient to maximize the logarithm of the
# PDF because it is a monotonically increasing function of the argument.
# Alternatively, and this will be our option, we will minimize the
# negative of the logarithm since this is a monotonically decreasing
# function.
# 
# Note also that maximization/minimization of the logarithm of the PDF
# is equivalent to the maximization/minimization of the function itself.

# ## A new Cost Function
# 
# We could now define a new cost function to minimize, namely the negative logarithm of the above PDF

# $$
# C(\boldsymbol{\beta}=-\log{\prod_{i=0}^{n-1}p(y_i,\boldsymbol{X}\vert\boldsymbol{\beta})}=-\sum_{i=0}^{n-1}\log{p(y_i,\boldsymbol{X}\vert\boldsymbol{\beta})},
# $$

# which becomes

# $$
# C(\boldsymbol{\beta}=\frac{n}{2}\log{2\pi\sigma^2}+\frac{\vert\vert (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\vert\vert_2^2}{2\sigma^2}.
# $$

# Taking the derivative of the *new* cost function with respect to the parameters $\beta$ we recognize our familiar OLS equation, namely

# $$
# \boldsymbol{X}^T\left(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right) =0,
# $$

# which leads to the well-known OLS equation for the optimal paramters $\beta$

# $$
# \hat{\boldsymbol{\beta}}^{\mathrm{OLS}}=\left(\boldsymbol{X}^T\boldsymbol{X}\right)^{-1}\boldsymbol{X}^T\boldsymbol{y}!
# $$

# Before we make a similar analysis for Ridge and Lasso regression, we need a short reminder on statistics.

# ## More basic Statistics and Bayes' theorem
# 
# A central theorem in statistics is Bayes' theorem. This theorem plays a similar role as the good old Pythagoras' theorem in geometry.
# Bayes' theorem is extremely simple to derive. But to do so we need some basic axioms from statistics.
# 
# Assume we have two domains of events $X=[x_0,x_1,\dots,x_{n-1}]$ and $Y=[y_0,y_1,\dots,y_{n-1}]$.
# 
# We define also the likelihood for $X$ and $Y$ as $p(X)$ and $p(Y)$ respectively.
# The likelihood of a specific event $x_i$ (or $y_i$) is then written as $p(X=x_i)$ or just $p(x_i)=p_i$. 
# 
# **Union of events is given by.**

# $$
# p(X \cup Y)= p(X)+p(Y)-p(X \cap Y).
# $$

# **The product rule (aka joint probability) is given by.**

# $$
# p(X \cup Y)= p(X,Y)= p(X\vert Y)p(Y)=p(Y\vert X)p(X),
# $$

# where we read $p(X\vert Y)$ as the likelihood of obtaining $X$ given $Y$.
# 
# If we have independent events then $p(X,Y)=p(X)p(Y)$.

# ## Marginal Probability
# 
# The marginal probability is defined in terms of only one of the set of variables $X,Y$. For a discrete probability we have

# $$
# p(X)=\sum_{i=0}^{n-1}p(X,Y=y_i)=\sum_{i=0}^{n-1}p(X\vert Y=y_i)p(Y=y_i)=\sum_{i=0}^{n-1}p(X\vert y_i)p(y_i).
# $$

# ## Conditional  Probability
# 
# The conditional  probability, if $p(Y) > 0$, is

# $$
# p(X\vert Y)= \frac{p(X,Y)}{p(Y)}=\frac{p(X,Y)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)}.
# $$

# ## Bayes' Theorem
# 
# If we combine the conditional probability with the marginal probability and the standard product rule, we have

# $$
# p(X\vert Y)= \frac{p(X,Y)}{p(Y)},
# $$

# which we can rewrite as

# $$
# p(X\vert Y)= \frac{p(X,Y)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)}=\frac{p(Y\vert X)p(X)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)},
# $$

# which is Bayes' theorem. It allows us to evaluate the uncertainty in in $X$ after we have observed $Y$. We can easily interchange $X$ with $Y$.

# ## Interpretations of Bayes' Theorem
# 
# The quantity $p(Y\vert X)$ on the right-hand side of the theorem is
# evaluated for the observed data $Y$ and can be viewed as a function of
# the parameter space represented by $X$. This function is not
# necesseraly normalized and is normally called the likelihood function.
# 
# The function $p(X)$ on the right hand side is called the prior while the function on the left hand side is the called the posterior probability. The denominator on the right hand side serves as a normalization factor for the posterior distribution.
# 
# Let us try to illustrate Bayes' theorem through an example.

# ## Example of Usage of Bayes' theorem
# 
# Let us suppose that you are undergoing a series of mammography scans in
# order to rule out possible breast cancer cases.  We define the
# sensitivity for a positive event by the variable $X$. It takes binary
# values with $X=1$ representing a positive event and $X=0$ being a
# negative event. We reserve $Y$ as a classification parameter for
# either a negative or a positive breast cancer confirmation. (Short note on wordings: positive here means having breast cancer, although none of us would consider this being a  positive thing).
# 
# We let $Y=1$ represent the the case of having breast cancer and $Y=0$ as not.
# 
# Let us assume that if you have breast cancer, the test will be positive with a probability of $0.8$, that is we have

# $$
# p(X=1\vert Y=1) =0.8.
# $$

# This obviously sounds  scary since many would conclude that if the test is positive, there is a likelihood of $80\%$ for having cancer.
# It is however not correct, as the following Bayesian analysis shows.

# ## Doing it correctly
# 
# If we look at various national surveys on breast cancer, the general likelihood of developing breast cancer is a very small number.
# Let us assume that the prior probability in the population as a whole is

# $$
# p(Y=1) =0.004.
# $$

# We need also to account for the fact that the test may produce a false positive result (false alarm). Let us here assume that we have

# $$
# p(X=1\vert Y=0) =0.1.
# $$

# Using Bayes' theorem we can then find the posterior probability that the person has breast cancer in case of a positive test, that is we can compute

# $$
# p(Y=1\vert X=1)=\frac{p(X=1\vert Y=1)p(Y=1)}{p(X=1\vert Y=1)p(Y=1)+p(X=1\vert Y=0)p(Y=0)}=\frac{0.8\times 0.004}{0.8\times 0.004+0.1\times 0.996}=0.031.
# $$

# That is, in case of a positive test, there is only a $3\%$ chance of having breast cancer!

# ## Bayes' Theorem and Ridge and Lasso Regression
# 
# Hitherto we have discussed Ridge and Lasso regression in terms of a
# linear analysis. This may to many of you feel rather technical and
# perhaps not that intuitive. The question is whether we can develop a
# more intuitive way of understanding what Ridge and Lasso express.
# 
# Before we proceed let us perform a Ridge, Lasso  and OLS analysis of a polynomial fit.

# ## Test Function for what happens with OLS, Ridge and Lasso
# 
# We will play around with a study of the values for the optimal
# parameters $\boldsymbol{\beta}$ using OLS, Ridge and Lasso regression.  For
# OLS, you will notice as function of the noise and polynomial degree,
# that the parameters $\beta$ will fluctuate from order to order in the
# polynomial fit and that for larger and larger polynomial degrees of freedom, the parameters will tend to increase in value for OLS.
# 
# For Ridge and Lasso regression, the higher order parameters will typically be reduced, providing thereby less fluctuations from one order to another one.

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# Make data set.
n = 10000
x = np.random.rand(n)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.randn(n)

Maxpolydegree = 5
X = np.zeros((len(x),Maxpolydegree))
X[:,0] = 1.0

for polydegree in range(1, Maxpolydegree):
    for degree in range(polydegree):
        X[:,degree] = x**(degree)


# We split the data in test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# matrix inversion to find beta
OLSbeta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
print(OLSbeta)
ypredictOLS = X_test @ OLSbeta
print("Test MSE OLS")
print(MSE(y_test,ypredictOLS))
# Repeat now for Lasso and Ridge regression and various values of the regularization parameter using Scikit-Learn
# Decide which values of lambda to use
nlambdas = 4
MSERidgePredict = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
lambdas = np.logspace(-3, 1, nlambdas)
for i in range(nlambdas):
    lmb = lambdas[i]
    # Make the fit using Ridge and Lasso
    RegRidge = linear_model.Ridge(lmb,fit_intercept=False)
    RegRidge.fit(X_train,y_train)
    RegLasso = linear_model.Lasso(lmb,fit_intercept=False)
    RegLasso.fit(X_train,y_train)
    # and then make the prediction
    ypredictRidge = RegRidge.predict(X_test)
    ypredictLasso = RegLasso.predict(X_test)
    # Compute the MSE and print it
    MSERidgePredict[i] = MSE(y_test,ypredictRidge)
    MSELassoPredict[i] = MSE(y_test,ypredictLasso)
    print(lmb,RegRidge.coef_)
    print(lmb,RegLasso.coef_)
# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSERidgePredict, 'b', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSELassoPredict, 'r', label = 'MSE Lasso Test')
plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()


# How can we understand this?

# ## Invoking Bayes' theorem
# 
# Using Bayes' theorem we can gain a better intuition about Ridge and Lasso regression. 
# 
# For ordinary least squares we postulated that the maximum likelihood for the doamin of events $\boldsymbol{D}$ (one-dimensional case)

# $$
# \boldsymbol{D}=[(x_0,y_0), (x_1,y_1),\dots, (x_{n-1},y_{n-1})],
# $$

# is given by

# $$
# p(\boldsymbol{D}\vert\boldsymbol{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}.
# $$

# In Bayes' theorem this function plays the role of the so-called likelihood. We could now ask the question what is the posterior probability of a parameter set $\boldsymbol{\beta}$ given a domain of events $\boldsymbol{D}$?  That is, how can we define the posterior probability

# $$
# p(\boldsymbol{\beta}\vert\boldsymbol{D}).
# $$

# Bayes' theorem comes to our rescue here since (omitting the normalization constant)

# $$
# p(\boldsymbol{\beta}\vert\boldsymbol{D})\propto p(\boldsymbol{D}\vert\boldsymbol{\beta})p(\boldsymbol{\beta}).
# $$

# We have a model for $p(\boldsymbol{D}\vert\boldsymbol{\beta})$ but need one for the **prior** $p(\boldsymbol{\beta}$!

# ## Ridge and Bayes
# 
# With the posterior probability defined by a likelihood which we have
# already modeled and an unknown prior, we are now ready to make
# additional models for the prior.
# 
# We can, based on our discussions of the variance of $\boldsymbol{\beta}$ and the mean value, assume that the prior for the values $\boldsymbol{\beta}$ is given by a Gaussian with mean value zero and variance $\tau^2$, that is

# $$
# p(\boldsymbol{\beta})=\prod_{j=0}^{p-1}\exp{\left(-\frac{\beta_j^2}{2\tau^2}\right)}.
# $$

# Our posterior probability becomes then (omitting the normalization factor which is just a constant)

# $$
# p(\boldsymbol{\beta\vert\boldsymbol{D})}=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}\prod_{j=0}^{p-1}\exp{\left(-\frac{\beta_j^2}{2\tau^2}\right)}.
# $$

# We can now optimize this quantity with respect to $\boldsymbol{\beta}$. As we
# did for OLS, this is most conveniently done by taking the negative
# logarithm of the posterior probability. Doing so and leaving out the
# constants terms that do not depend on $\beta$, we have

# $$
# C(\boldsymbol{\beta})=\frac{\vert\vert (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\vert\vert_2^2}{2\sigma^2}+\frac{1}{2\tau^2}\vert\vert\boldsymbol{\beta}\vert\vert_2^2,
# $$

# and replacing $1/2\tau^2$ with $\lambda$ we have

# $$
# C(\boldsymbol{\beta})=\frac{\vert\vert (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\vert\vert_2^2}{2\sigma^2}+\lambda\vert\vert\boldsymbol{\beta}\vert\vert_2^2,
# $$

# which is our Ridge cost function!  Nice, isn't it?

# ## Lasso and Bayes
# 
# To derive the Lasso cost function, we simply replace the Gaussian prior with an exponential distribution ([Laplace in this case](https://en.wikipedia.org/wiki/Laplace_distribution)) with zero mean value,  that is

# $$
# p(\boldsymbol{\beta})=\prod_{j=0}^{p-1}\exp{\left(-\frac{\vert\beta_j\vert}{\tau}\right)}.
# $$

# Our posterior probability becomes then (omitting the normalization factor which is just a constant)

# $$
# p(\boldsymbol{\beta}\vert\boldsymbol{D})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\boldsymbol{X}_{i,*}\boldsymbol{\beta})^2}{2\sigma^2}\right]}\prod_{j=0}^{p-1}\exp{\left(-\frac{\vert\beta_j\vert}{\tau}\right)}.
# $$

# Taking the negative
# logarithm of the posterior probability and leaving out the
# constants terms that do not depend on $\beta$, we have

# $$
# C(\boldsymbol{\beta}=\frac{\vert\vert (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\vert\vert_2^2}{2\sigma^2}+\frac{1}{\tau}\vert\vert\boldsymbol{\beta}\vert\vert_1,
# $$

# and replacing $1/\tau$ with $\lambda$ we have

# $$
# C(\boldsymbol{\beta}=\frac{\vert\vert (\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta})\vert\vert_2^2}{2\sigma^2}+\lambda\vert\vert\boldsymbol{\beta}\vert\vert_1,
# $$

# which is our Lasso cost function!
