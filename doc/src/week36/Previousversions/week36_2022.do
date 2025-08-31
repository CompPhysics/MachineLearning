TITLE: Week 36: Statistical interpretation of Linear Regression and Resampling techniques
AUTHOR: Morten Hjorth-Jensen {copyright, 1999-present|CC BY-NC} at Department of Physics, University of Oslo & Department of Physics and Astronomy and National Superconducting Cyclotron Laboratory, Michigan State University
DATE: today


!split
===== Plans for week 36 =====

* Summary from last week on discussion of SVD, Ridge and Lasso linear regression.
* Linear Regression and links with Statistics, Resampling methods and presentation of first project

Recommended Reading:
o Lectures on Regression
o Bishop 1.1, 1.2, 2.1, 2.2, 2.3 and 3.1
o Hastie et al chapter 3



!split
===== Summary from last Week and discussion of SVD, Ridge and Lasso regression with examples =====

!split
===== Linear Regression and  the SVD =====

We used the SVD to analyse the matrix to invert in ordinary lineat regression
!bt
\[
\bm{X}^T\bm{X}=\bm{V}\bm{\Sigma}^T\bm{U}^T\bm{U}\bm{\Sigma}\bm{V}^T=\bm{V}\bm{\Sigma}^T\bm{\Sigma}\bm{V}^T. 
\]
!et
Since the matrices here have dimension $p\times p$, with $p$ corresponding to the singular values, we defined last week the matrix
!bt
\[
\bm{\Sigma}^T\bm{\Sigma} = \begin{bmatrix} \tilde{\bm{\Sigma}} & \bm{0}\\ \end{bmatrix}\begin{bmatrix} \tilde{\bm{\Sigma}} \\ \bm{0}\end{bmatrix},
\]
!et
where the tilde-matrix $\tilde{\bm{\Sigma}}$ is a matrix of dimension $p\times p$ containing only the singular values $\sigma_i$, that is

!bt
\[
\tilde{\bm{\Sigma}}=\begin{bmatrix} \sigma_0 & 0 & 0 & \dots & 0 & 0 \\
                                    0 & \sigma_1 & 0 & \dots & 0 & 0 \\
				    0 & 0 & \sigma_2 & \dots & 0 & 0 \\
				    0 & 0 & 0 & \dots & \sigma_{p-2} & 0 \\
				    0 & 0 & 0 & \dots & 0 & \sigma_{p-1} \\
\end{bmatrix},
\]
!et
meaning we can write
!bt
\[
\bm{X}^T\bm{X}=\bm{V}\tilde{\bm{\Sigma}}^2\bm{V}^T. 
\]
!et
Multiplying from the right with $\bm{V}$ (using the orthogonality of $\bm{V}$) we get
!bt
\[
\left(\bm{X}^T\bm{X}\right)\bm{V}=\bm{V}\tilde{\bm{\Sigma}}^2. 
\]
!et

!split
===== What does it mean? =====

This means the vectors $\bm{v}_i$ of the orthogonal matrix $\bm{V}$
are the eigenvectors of the matrix $\bm{X}^T\bm{X}$ with eigenvalues
given by the singular values squared, that is

!bt
\[
\left(\bm{X}^T\bm{X}\right)\bm{v}_i=\bm{v}_i\sigma_i^2. 
\]
!et

In other words, each non-zero singular value of $\bm{X}$ is a positive
square root of an eigenvalue of $\bm{X}^T\bm{X}$.  It means also that
the columns of $\bm{V}$ are the eigenvectors of
$\bm{X}^T\bm{X}$. Since we have ordered the singular values of
$\bm{X}$ in a descending order, it means that the column vectors
$\bm{v}_i$ are hierarchically ordered by how much correlation they
encode from the columns of $\bm{X}$. 


Note that these are also the eigenvectors and eigenvalues of the
Hessian matrix.

If we now recall the definition of the covariance matrix (not using
Bessel's correction) we have


!bt
\[
\bm{C}[\bm{X}]=\frac{1}{n}\bm{X}^T\bm{X},
\]
!et

meaning that every squared non-singular value of $\bm{X}$ divided by $n$ (
the number of samples) are the eigenvalues of the covariance
matrix. Every singular value of $\bm{X}$ is thus a positive square
root of an eigenvalue of $\bm{X}^T\bm{X}$. If the matrix $\bm{X}$ is
self-adjoint, the singular values of $\bm{X}$ are equal to the
absolute value of the eigenvalues of $\bm{X}$.

!split
===== And finally  $\bm{X}\bm{X}^T$ =====

For $\bm{X}\bm{X}^T$ we found

!bt
\[
\bm{X}\bm{X}^T=\bm{U}\bm{\Sigma}\bm{V}^T\bm{V}\bm{\Sigma}^T\bm{U}^T=\bm{U}\bm{\Sigma}^T\bm{\Sigma}\bm{U}^T. 
\]
!et
Since the matrices here have dimension $n\times n$, we have
!bt
\[
\bm{\Sigma}\bm{\Sigma}^T = \begin{bmatrix} \tilde{\bm{\Sigma}} \\ \bm{0}\\ \end{bmatrix}\begin{bmatrix} \tilde{\bm{\Sigma}}  \bm{0}\\ \end{bmatrix}=\begin{bmatrix} \tilde{\bm{\Sigma}} & \bm{0} \\ \bm{0} & \bm{0}\\ \end{bmatrix}, 
\]
!et
leading to
!bt
\[
\bm{X}\bm{X}^T=\bm{U}\begin{bmatrix} \tilde{\bm{\Sigma}} & \bm{0} \\ \bm{0} & \bm{0}\\ \end{bmatrix}\bm{U}^T. 
\]
!et

Multiplying with $\bm{U}$ from the right gives us the eigenvalue problem
!bt
\[
(\bm{X}\bm{X}^T)\bm{U}=\bm{U}\begin{bmatrix} \tilde{\bm{\Sigma}} & \bm{0} \\ \bm{0} & \bm{0}\\ \end{bmatrix}. 
\]
!et

It means that the eigenvalues of $\bm{X}\bm{X}^T$ are again given by
the non-zero singular values plus now a series of zeros.  The column
vectors of $\bm{U}$ are the eigenvectors of $\bm{X}\bm{X}^T$ and
measure how much correlations are contained in the rows of $\bm{X}$.

Since we will mainly be interested in the correlations among the features
of our data (the columns of $\bm{X}$, the quantity of interest for us are the non-zero singular
values and the column vectors of $\bm{V}$.



!split
=====  Code for SVD and Inversion of Matrices =====

How do we use the SVD to invert a matrix $\bm{X}^\bm{X}$ which is singular or near singular?
The simple answer is to use the linear algebra function for pseudoinvers, that is
!bc pycod
Ainv = np.linlag.pinv(A)
!ec

Let us first look at a matrix which does not causes problems and write our own function where we just use the SVD.

!bc pycod
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

!ec

!split
===== Inverse of Rectangular Matrix =====

Although our matrix to invert $\bm{X}^T\bm{X}$ is a square matrix, our matrix may be singular. 

The pseudoinverse is the generalization of the matrix inverse for square matrices to
rectangular matrices where the number of rows and columns are not equal.

It is also called the the Moore-Penrose Inverse after two independent discoverers of the method or the Generalized Inverse.
It is used for the calculation of the inverse for singular or near singular matrices and for rectangular matrices.

Using the SVD we can obtain the pseudoinverse of a matrix $\bm{A}$ (labeled here as $\bm{A}_{\mathrm{PI}}$)
!bt
\[
\bm{A}_{\mathrm{PI}}= \bm{V}\bm{D}_{\mathrm{PI}}\bm{U}^T,
\]
!et
where $\bm{D}_{\mathrm{PI}}$ can be calculated by creating a diagonal matrix from $\bm{\Sigma}$ where we only keep the singular values (the non-zero values). The following code computes the pseudoinvers of the matrix based on the SVD.


!bc pycod
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

!ec
As you can see from this example, our own decomposition based on the SVD agrees the pseudoinverse algorithm provided by _Numpy_.



!split
===== Ridge and LASSO Regression =====

Let us remind ourselves about the expression for the standard Mean Squared Error (MSE) which we used to define our cost function and the equations for the ordinary least squares (OLS) method, that is 
our optimization problem is
!bt
\[
{\displaystyle \min_{\bm{\beta}\in {\mathbb{R}}^{p}}}\frac{1}{n}\left\{\left(\bm{y}-\bm{X}\bm{\beta}\right)^T\left(\bm{y}-\bm{X}\bm{\beta}\right)\right\}.
\]
!et
or we can state it as
!bt
\[
{\displaystyle \min_{\bm{\beta}\in
{\mathbb{R}}^{p}}}\frac{1}{n}\sum_{i=0}^{n-1}\left(y_i-\tilde{y}_i\right)^2=\frac{1}{n}\vert\vert \bm{y}-\bm{X}\bm{\beta}\vert\vert_2^2,
\]
!et
where we have used the definition of  a norm-2 vector, that is
!bt
\[
\vert\vert \bm{x}\vert\vert_2 = \sqrt{\sum_i x_i^2}. 
\]
!et

!split
===== From OLS to Ridge and Lasso =====

By minimizing the above equation with respect to the parameters
$\bm{\beta}$ we could then obtain an analytical expression for the
parameters $\bm{\beta}$.  We can add a regularization parameter $\lambda$ by
defining a new cost function to be optimized, that is

!bt
\[
{\displaystyle \min_{\bm{\beta}\in
{\mathbb{R}}^{p}}}\frac{1}{n}\vert\vert \bm{y}-\bm{X}\bm{\beta}\vert\vert_2^2+\lambda\vert\vert \bm{\beta}\vert\vert_2^2
\]
!et

which leads to the Ridge regression minimization problem where we
require that $\vert\vert \bm{\beta}\vert\vert_2^2\le t$, where $t$ is
a finite number larger than zero. By defining

!bt
\[
C(\bm{X},\bm{\beta})=\frac{1}{n}\vert\vert \bm{y}-\bm{X}\bm{\beta}\vert\vert_2^2+\lambda\vert\vert \bm{\beta}\vert\vert_1,
\]
!et

we have a new optimization equation
!bt
\[
{\displaystyle \min_{\bm{\beta}\in
{\mathbb{R}}^{p}}}\frac{1}{n}\vert\vert \bm{y}-\bm{X}\bm{\beta}\vert\vert_2^2+\lambda\vert\vert \bm{\beta}\vert\vert_1
\]
!et
which leads to Lasso regression. Lasso stands for least absolute shrinkage and selection operator. 

Here we have defined the norm-1 as 
!bt
\[
\vert\vert \bm{x}\vert\vert_1 = \sum_i \vert x_i\vert. 
\]
!et


!split
===== Deriving the  Ridge Regression Equations =====

Using the matrix-vector expression for Ridge regression and dropping the parameter $1/n$ in front of the standard means squared error equation, we have

!bt
\[
C(\bm{X},\bm{\beta})=\left\{(\bm{y}-\bm{X}\bm{\beta})^T(\bm{y}-\bm{X}\bm{\beta})\right\}+\lambda\bm{\beta}^T\bm{\beta},
\]
!et
and 
taking the derivatives with respect to $\bm{\beta}$ we obtain then
a slightly modified matrix inversion problem which for finite values
of $\lambda$ does not suffer from singularity problems. We obtain
the optimal parameters
!bt
\[
\hat{\bm{\beta}}_{\mathrm{Ridge}} = \left(\bm{X}^T\bm{X}+\lambda\bm{I}\right)^{-1}\bm{X}^T\bm{y},
\]
!et

with $\bm{I}$ being a $p\times p$ identity matrix with the constraint that



!bt
\[
\sum_{i=0}^{p-1} \beta_i^2 \leq t,
\]
!et

with $t$ a finite positive number. 


!split
===== Note on Scikit-Learn =====

Note well that a library like _Scikit-Learn_ does not include the $1/n$ factor in the expression for the mean-squared error. If you include it, the optimal parameter $\beta$ becomes

!bt
\[
\hat{\bm{\beta}}_{\mathrm{Ridge}} = \left(\bm{X}^T\bm{X}+n\lambda\bm{I}\right)^{-1}\bm{X}^T\bm{y}.
\]
!et

In our codes where we compare our own codes with _Scikit-Learn_, we do thus not include the $1/n$ factor in the cost function.


!split
===== Comparison with OLS =====
When we compare this with the ordinary least squares result we have
!bt
\[
\hat{\bm{\beta}}_{\mathrm{OLS}} = \left(\bm{X}^T\bm{X}\right)^{-1}\bm{X}^T\bm{y},
\]
!et
which can lead to singular matrices. However, with the SVD, we can always compute the inverse of the matrix $\bm{X}^T\bm{X}$.


We see that Ridge regression is nothing but the standard OLS with a
modified diagonal term added to $\bm{X}^T\bm{X}$. The consequences, in
particular for our discussion of the bias-variance tradeoff are rather
interesting. We will see that for specific values of $\lambda$, we may
even reduce the variance of the optimal parameters $\bm{\beta}$. These topics and other related ones, will be discussed after the more linear algebra oriented analysis here.

!split
===== SVD analysis =====

Using our insights about the SVD of the design matrix $\bm{X}$ 
We have already analyzed the OLS solutions in terms of the eigenvectors (the columns) of the right singular value matrix $\bm{U}$ as
!bt
\[
\tilde{\bm{y}}_{\mathrm{OLS}}=\bm{X}\bm{\beta}  =\bm{U}\bm{U}^T\bm{y}.
\]
!et


For Ridge regression this becomes

!bt
\[
\tilde{\bm{y}}_{\mathrm{Ridge}}=\bm{X}\bm{\beta}_{\mathrm{Ridge}} = \bm{U\Sigma V^T}\left(\bm{V}\bm{\Sigma}^2\bm{V}^T+\lambda\bm{I} \right)^{-1}(\bm{U\Sigma V^T})^T\bm{y}=\sum_{j=0}^{p-1}\bm{u}_j\bm{u}_j^T\frac{\sigma_j^2}{\sigma_j^2+\lambda}\bm{y},
\]
!et

with the vectors $\bm{u}_j$ being the columns of $\bm{U}$ from the SVD of the matrix $\bm{X}$. 

!split
===== Interpreting the Ridge results =====

Since $\lambda \geq 0$, it means that compared to OLS, we have 

!bt
\[
\frac{\sigma_j^2}{\sigma_j^2+\lambda} \leq 1. 
\]
!et

Ridge regression finds the coordinates of $\bm{y}$ with respect to the
orthonormal basis $\bm{U}$, it then shrinks the coordinates by
$\frac{\sigma_j^2}{\sigma_j^2+\lambda}$. Recall that the SVD has
eigenvalues ordered in a descending way, that is $\sigma_i \geq
\sigma_{i+1}$.

For small eigenvalues $\sigma_i$ it means that their contributions become less important, a fact which can be used to reduce the number of degrees of freedom. More about this when we have covered the material on a statistical interpretation of various linear regression methods.


!split
===== More interpretations =====

For the sake of simplicity, let us assume that the design matrix is orthonormal, that is 

!bt
\[
\bm{X}^T\bm{X}=(\bm{X}^T\bm{X})^{-1} =\bm{I}. 
\]
!et

In this case the standard OLS results in 
!bt
\[
\bm{\beta}^{\mathrm{OLS}} = \bm{X}^T\bm{y}=\sum_{i=0}^{n-1}\bm{u}_i\bm{u}_i^T\bm{y},
\]
!et

and

!bt
\[
\bm{\beta}^{\mathrm{Ridge}} = \left(\bm{I}+\lambda\bm{I}\right)^{-1}\bm{X}^T\bm{y}=\left(1+\lambda\right)^{-1}\bm{\beta}^{\mathrm{OLS}},
\]
!et

that is the Ridge estimator scales the OLS estimator by the inverse of a factor $1+\lambda$, and
the Ridge estimator converges to zero when the hyperparameter goes to
infinity.

We will come back to more interpreations after we have gone through some of the statistical analysis part. 

For more discussions of Ridge and Lasso regression, "Wessel van Wieringen's":"https://arxiv.org/abs/1509.09169" article is highly recommended.
Similarly, "Mehta et al's article":"https://arxiv.org/abs/1803.08823" is also recommended.

!split
===== Deriving the  Lasso Regression Equations =====

Using the matrix-vector expression for Lasso regression and dropping the parameter $1/n$ in front of the standard mean squared error equation, we have the following _cost_ function

!bt
\[
C(\bm{X},\bm{\beta})=\left\{(\bm{y}-\bm{X}\bm{\beta})^T(\bm{y}-\bm{X}\bm{\beta})\right\}+\lambda\vert\vert\bm{\beta}\vert\vert_1,
\]
!et

Taking the derivative with respect to $\bm{\beta}$ and recalling that the derivative of the absolute value is (we drop the boldfaced vector symbol for simplicty)
!bt
\[
\frac{d \vert \beta\vert}{d \bm{\beta}}=\mathrm{sgn}(\bm{\beta})=\left\{\begin{array}{cc} 1 & \beta > 0 \\-1 & \beta < 0, \end{array}\right.
\]
!et
we have that the derivative of the cost function is

!bt
\[
\frac{\partial C(\bm{X},\bm{\beta})}{\partial \bm{\beta}}=-2\bm{X}^T(\bm{y}-\bm{X}\bm{\beta})+\lambda sgn(\bm{\beta})=0,
\]
!et
and reordering we have
!bt
\[
\bm{X}^T\bm{X}\bm{\beta}+\lambda sgn(\bm{\beta})=2\bm{X}^T\bm{y}.
\]
!et
This equation does not lead to a nice analytical equation as in Ridge regression or ordinary least squares. This equation can however be solved by using standard convex optimization algorithms using for example the Python package "CVXOPT":"https://cvxopt.org/". We will discuss this later. 




!split
===== Simple example to illustrate Ordinary Least Squares, Ridge and Lasso Regression =====

Let us assume that our design matrix is given by unit (identity) matrix, that is a square diagonal matrix with ones only along the
diagonal. In this case we have an equal number of rows and columns $n=p$.

Our model approximation is just $\tilde{\bm{y}}=\bm{\beta}$ and the mean squared error and thereby the cost function for ordinary least sqquares (OLS) is then (we drop the term $1/n$) 
!bt
\[
C(\bm{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2,
\]
!et
and minimizing we have that
!bt
\[
\hat{\beta}_i^{\mathrm{OLS}} = y_i.
\]
!et

!split
===== Ridge Regression =====

For Ridge regression our cost function is
!bt
\[
C(\bm{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\beta_i^2,
\]
!et
and minimizing we have that
!bt
\[
\hat{\beta}_i^{\mathrm{Ridge}} = \frac{y_i}{1+\lambda}.
\]
!et


!split
===== Lasso Regression =====

For Lasso regression our cost function is
!bt
\[
C(\bm{\beta})=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\vert\beta_i\vert=\sum_{i=0}^{p-1}(y_i-\beta_i)^2+\lambda\sum_{i=0}^{p-1}\sqrt{\beta_i^2},
\]
!et
and minimizing we have that
!bt
\[
-2\sum_{i=0}^{p-1}(y_i-\beta_i)+\lambda \sum_{i=0}^{p-1}\frac{(\beta_i)}{\vert\beta_i\vert}=0,
\]
!et
which leads to 
!bt
\[
\hat{\bm{\beta}}_i^{\mathrm{Lasso}} = \left\{\begin{array}{ccc}y_i-\frac{\lambda}{2} &\mathrm{if} & y_i> \frac{\lambda}{2}\\
                                                          y_i+\frac{\lambda}{2} &\mathrm{if} & y_i< -\frac{\lambda}{2}\\
							  0 &\mathrm{if} & \vert y_i\vert\le  \frac{\lambda}{2}\end{array}\right.\\.
\]
!et

Plotting these results ("figure in handwritten notes for week 36":"https://github.com/CompPhysics/MachineLearning/blob/master/doc/HandWrittenNotes/2021/NotesSeptember9.pdf") shows clearly that Lasso regression suppresses (sets to zero) values of $\beta_i$ for specific values of $\lambda$. Ridge regression reduces on the other hand the values of $\beta_i$ as function of $\lambda$.



!split
===== Yet another Example =====

Let us assume we have a data set with outputs/targets given by the vector

!bt
\[
\bm{y}=\begin{bmatrix}4 \\ 2 \\3\end{bmatrix},
\]
!et
and our inputs as a $3\times 2$ design matrix
!bt
\[
\bm{X}=\begin{bmatrix}2 & 0\\ 0 & 1 \\ 0 & 0\end{bmatrix},
\]
!et
meaning that we have two features and two unknown parameters $\beta_0$ and $\beta_1$ to be determined either by ordinary least squares, Ridge or Lasso regression.

!split
===== The OLS case =====

For ordinary least squares (OLS) we know that the optimal solution is

!bt
\[
\hat{\bm{\beta}}^{\mathrm{OLS}}=\left( \bm{X}^T\bm{X}\right)^{-1}\bm{X}^T\bm{y}.
\]
!et
Inserting the above values we obtain that 

!bt
\[
\hat{\bm{\beta}}^{\mathrm{OLS}}=\begin{bmatrix}2 \\ 2\end{bmatrix},
\]
!et

The code which implements this simpler case is presented after the discussion of Ridge and Lasso.

!split
===== The Ridge case =====

For Ridge regression we have

!bt
\[
\hat{\bm{\beta}}^{\mathrm{Ridge}}=\left( \bm{X}^T\bm{X}+\lambda\bm{I}\right)^{-1}\bm{X}^T\bm{y}.
\]
!et
Inserting the above values we obtain that 

!bt
\[
\hat{\bm{\beta}}^{\mathrm{Ridge}}=\begin{bmatrix}\frac{8}{4+\lambda} \\ \frac{2}{1+\lambda}\end{bmatrix},
\]
!et

There is normally a constraint on the value of $\vert\vert \bm{\beta}\vert\vert_2$ via the parameter $\lambda$.
Let us for simplicity assume that $\beta_0^2+\beta_1^2=1$ as constraint. This will allow us to find an expression for the optimal values of $\beta$ and $\lambda$.

To see this, let us write the cost function for Ridge regression.  


!split
===== Writing the Cost Function =====

We define the MSE without the $1/n$ factor and have then, using that
!bt
\[
\bm{X}\bm{\beta}=\begin{bmatrix} 2\beta_0 \\ \beta_1 \\0 \end{bmatrix},
\]
!et

!bt
\[
C(\bm{\beta})=(4-2\beta_0)^2+(2-\beta_1)^2+\lambda(\beta_0^2+\beta_1^2),
\]
!et
and taking the derivative with respect to $\beta_0$ we get
!bt
\[
\beta_0=\frac{8}{4+\lambda},
\]
!et
and for $\beta_1$ we obtain
!bt
\[
\beta_1=\frac{2}{1+\lambda},
\]
!et

Using the constraint for $\beta_0^2+\beta_1^2=1$ we can constrain $\lambda$ by solving
!bt
\[
\left(\frac{8}{4+\lambda}\right)^2+\left(\frac{2}{1+\lambda}\right)^2=1,
\]
!et
which gives $\lambda=4.571$ and $\beta_0=0.933$ and $\beta_1=0.359$.

!split
===== Lasso case =====

For Lasso we need now, keeping a  constraint on $\vert\beta_0\vert+\vert\beta_1\vert=1$,  to take the derivative of the absolute values of $\beta_0$
and $\beta_1$. This gives us the following derivatives of the cost function
!bt
\[
C(\bm{\beta})=(4-2\beta_0)^2+(2-\beta_1)^2+\lambda(\vert\beta_0\vert+\vert\beta_1\vert),
\]
!et

!bt
\[
\frac{\partial C(\bm{\beta})}{\partial \beta_0}=-4(4-2\beta_0)+\lambda\mathrm{sgn}(\beta_0)=0,
\]
!et
and
!bt
\[
\frac{\partial C(\bm{\beta})}{\partial \beta_1}=-2(2-\beta_1)+\lambda\mathrm{sgn}(\beta_1)=0.
\]
!et
We have now four cases to solve besides the trivial cases $\beta_0$ and/or $\beta_1$ are zero, namely
o $\beta_0 > 0$ and $\beta_1 > 0$,
o $\beta_0 > 0$ and $\beta_1 < 0$,
o $\beta_0 < 0$ and $\beta_1 > 0$,
o $\beta_0 < 0$ and $\beta_1 < 0$.

!split
===== The first Case =====

If we consider the first case, we have then
!bt
\[
-4(4-2\beta_0)+\lambda=0,
\]
!et
and
!bt
\[
-2(2-\beta_1)+\lambda=0.
\]
!et
which yields

!bt
\[
\beta_0=\frac{16+\lambda}{8},
\]
!et
and
!bt
\[
\beta_1=\frac{4+\lambda}{2}.
\]
!et

Using the constraint on $\beta_0$ and $\beta_1$ we can then find the optimal value of $\lambda$ for the different cases. We leave this as an exercise to you.

!split
===== Simple code for solving the above problem =====

Here we set up the OLS, Ridge and Lasso functionality in order to study the above example. Note that here we have opted for a set of values of $\lambda$, meaning that we need to perform a search in order to find the optimal values.

First we study and compare the OLS and Ridge results.  The next code compares all three methods.


!bc pycod
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

!ec

We see here that we reach a plateau. What is actually happening?


!split
===== With Lasso Regression =====

!bc pycod
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
    RegLasso = linear_model.Lasso(lmb)
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

!ec

!split
=====  Another Example, now with a polynomial fit =====

!bc pycod
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
    RegLasso = linear_model.Lasso(lmb)
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

!ec



!split
===== To think about, first part =====

When you are comparing your own code with for example _Scikit-Learn_'s
library, there are some technicalities to keep in mind.  The examples
here demonstrate some of these aspects with potential pitfalls.

The discussion here focuses on the role of the intercept, how we can
set up the design matrix, what scaling we should use and other topics
which tend  confuse us.



The intercept can be interpreted as the expected value of our
target/output variables when all other predictors are set to zero.
Thus, if we cannot assume that the expected outputs/targets are zero
when all predictors are zero (the columns in the design matrix), it
may be a bad idea to implement a model which penalizes the intercept.
Furthermore, in for example Ridge and Lasso regression, the default solutions
from the library _Scikit-Learn_ (when not shrinking $\beta_0$) for the unknown parameters
$\bm{\beta}$, are derived under the assumption that both $\bm{y}$ and
$\bm{X}$ are zero centered, that is we subtract the mean values.


!split
=====  More thinking =====


If our predictors represent different scales, then it is important to
standardize the design matrix $\bm{X}$ by subtracting the mean of each
column from the corresponding column and dividing the column with its
standard deviation. Most machine learning libraries do this as a default. This means that if you compare your code with the results from a given library,
the results may differ. 

The
"Standadscaler":"https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html"
function in _Scikit-Learn_ does this for us.  For the data sets we
have been studying in our various examples, the data are in many cases
already scaled and there is no need to scale them. You as a user of different machine learning algorithms, should always perform  a
survey of your data, with a critical assessment of them in case you need to scale the data.

If you need to scale the data, not doing so will give an *unfair*
penalization of the parameters since their magnitude depends on the
scale of their corresponding predictor.

Suppose as an example that you 
you have an input variable given by the heights of different persons.
Human height might be measured in inches or meters or
kilometers. If measured in kilometers, a standard linear regression
model with this predictor would probably give a much bigger
coefficient term, than if measured in millimeters.
This can clearly lead to problems in evaluating the cost/loss functions.


!split
===== Still thinking =====

Keep in mind that when you transform your data set before training a model, the same transformation needs to be done
on your eventual new data set  before making a prediction. If we translate this into a Python code, it would could be implemented as follows

!bc pycod
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
!ec


!split
===== What does centering (subtracting the mean values) mean mathematically? =====


Let us try to understand what this may imply mathematically when we
subtract the mean values, also known as *zero centering*. For
simplicity, we will focus on  ordinary regression, as done in the above example.

The cost/loss function  for regression is
!bt
\[
C(\beta_0, \beta_1, ... , \beta_{p-1}) = \frac{1}{n}\sum_{i=0}^{n} \left(y_i - \beta_0 - \sum_{j=1}^{p-1} X_{ij}\beta_j\right)^2,.
\]
!et
Recall also that we use the squared value since this leads to an increase of the penalty for higher differences between predicted and output/target values.

What we have done is to single out the $\beta_0$ term in the definition of the mean squared error (MSE).
The design matrix
$X$ does in this case not contain any intercept column.
When we take the derivative with respect to $\beta_0$, we want the derivative to obey
!bt
\[
\frac{\partial C}{\partial \beta_j} = 0,
\]
!et

for all $j$. For $\beta_0$ we have

!bt
\[
\frac{\partial C}{\partial \beta_0} = -\frac{2}{n}\sum_{i=0}^{n-1} \left(y_i - \beta_0 - \sum_{j=1}^{p-1} X_{ij} \beta_j\right).
\]
!et
Multiplying away the constant $2/n$, we obtain
!bt
\[
\sum_{i=0}^{n-1} \beta_0 = \sum_{i=0}^{n-1}y_i - \sum_{i=0}^{n-1} \sum_{j=1}^{p-1} X_{ij} \beta_j.
\]
!et

!split
===== Further Manipulations =====


Let us special first to the case where we have only two parameters $\beta_0$ and $\beta_1$.
Our result for $\beta_0$ simplifies then to
!bt
\[
n\beta_0 = \sum_{i=0}^{n-1}y_i - \sum_{i=0}^{n-1} X_{i1} \beta_1.
\]
!et
We obtain then
!bt
\[
\beta_0 = \frac{1}{n}\sum_{i=0}^{n-1}y_i - \beta_1\frac{1}{n}\sum_{i=0}^{n-1} X_{i1}.
\]
!et
If we define
!bt
\[
\mu_1=\frac{1}{n}\sum_{i=0}^{n-1} (X_{i1},
\]
!et
and if we define the mean value of the outputs as
!bt
\[
\mu_y=\frac{1}{n}\sum_{i=0}^{n-1}y_i,
\]
!et
we have
!bt
\[
\beta_0 = \mu_y - \beta_1\mu_{1}.
\]
!et
In the general case, that is we have more parameters than $\beta_0$ and $\beta_1$, we have
!bt
\[
\beta_0 = \frac{1}{n}\sum_{i=0}^{n-1}y_i - \frac{1}{n}\sum_{i=0}^{n-1}\sum_{j=1}^{p-1} X_{ij}\beta_j.
\]
!et



Replacing $y_i$ with $y_i - y_i - \overline{\bm{y}}$ and centering also our design matrix results in a cost function (in vector-matrix disguise)
!bt
\[
C(\boldsymbol{\beta}) = (\boldsymbol{\tilde{y}} - \tilde{X}\boldsymbol{\beta})^T(\boldsymbol{\tilde{y}} - \tilde{X}\boldsymbol{\beta}). 
\]
!et

!split
===== Wrapping it up =====

If we minimize with respect to $\bm{\beta}$ we have then

!bt
\[
\hat{\bm{\beta}} = (\tilde{X}^T\tilde{X})^{-1}\tilde{X}^T\boldsymbol{\tilde{y}},
\]
!et

where $\boldsymbol{\tilde{y}} = \boldsymbol{y} - \overline{\bm{y}}$
and $\tilde{X}_{ij} = X_{ij} - \frac{1}{n}\sum_{k=0}^{n-1}X_{kj}$.

For Ridge regression we need to add $\lambda \boldsymbol{\beta}^T\boldsymbol{\beta}$ to the cost function and get then
!bt
\[
\hat{\bm{\beta}} = (\tilde{X}^T\tilde{X} + \lambda I)^{-1}\tilde{X}^T\boldsymbol{\tilde{y}}.
\]
!et

What does this mean? And why do we insist on all this? Let us look at some examples.



!split
=====  Linear Regression code, Intercept handling first =====

This code shows a simple first-order fit to a data set using the above transformed data, where we consider the role of the intercept first, by either excluding it or including it (*code example thanks to  Øyvind Sigmundson Schøyen*). Here our scaling of the data is done by subtracting the mean values only.
Note also that we do not split the data into training and test.

!bc pycod
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

!ec

The intercept is the value of our output/target variable
when all our features are zero and our function crosses the $y$-axis (for a one-dimensional case). 

Printing the MSE, we see first that both methods give the same MSE, as
they should.  However, when we move to for example Ridge regression,
the way we treat the intercept may give a larger or smaller MSE,
meaning that the MSE can be penalized by the value of the
intercept. Not including the intercept in the fit, means that the
regularization term does not include $\beta_0$. For different values
of $\lambda$, this may lead to differeing MSE values. 

To remind the reader, the regularization term, with the intercept in Ridge regression is given by
!bt
\[
\lambda \vert\vert \bm{\beta} \vert\vert_2^2 = \lambda \sum_{j=0}^{p-1}\beta_j^2,
\]
!et
but when we take out the intercept, this equation becomes
!bt
\[
\lambda \vert\vert \bm{\beta} \vert\vert_2^2 = \lambda \sum_{j=1}^{p-1}\beta_j^2.
\]
!et

For Lasso regression we have
!bt
\[
\lambda \vert\vert \bm{\beta} \vert\vert_1 = \lambda \sum_{j=1}^{p-1}\vert\beta_j\vert.
\]
!et

It means that, when scaling the design matrix and the outputs/targets, by subtracting the mean values, we have an optimization problem which is not penalized by the intercept. The MSE value can then be smaller since it focuses only on the remaining quantities. If we however bring back the intercept, we will get a MSE which then contains the intercept. 

!split
===== Code Examples =====

Armed with this wisdom, we attempt first to simply set the intercept equal to _False_ in our implementation of Ridge regression for our well-known  vanilla data set.

!bc pycod
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

!ec

The results here agree when we force _Scikit-Learn_'s Ridge function to include the first column in our design matrix.
We see that the results agree very well. Here we have thus explicitely included the intercept column in the design matrix.
What happens if we do not include the intercept in our fit?
Let us see how we can change this code by zero centering (thanks to Stian Bilek for inpouts here).

!split
===== Taking out the mean =====
!bc pycod
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

!ec

We see here, when compared to the code which includes explicitely the
intercept column, that our MSE value is actually smaller. This is
because the regularization term does not include the intercept value
$\beta_0$ in the fitting.  This applies to Lasso regularization as
well.  It means that our optimization is now done only with the
centered matrix and/or vector that enter the fitting procedure. Note
also that the problem with the intercept occurs mainly in these type
of polynomial fitting problem.




!split
===== Friday September 9  =====



!split 
===== Linking the regression analysis with a statistical interpretation =====

We will now couple the discussions of ordinary least squares, Ridge
and Lasso regression with a statistical interpretation, that is we
move from a linear algebra analysis to a statistical analysis. In
particular, we will focus on what the regularization terms can result
in.  We will amongst other things show that the regularization
parameter can reduce considerably the variance of the parameters
$\beta$.


The
advantage of doing linear regression is that we actually end up with
analytical expressions for several statistical quantities.  
Standard least squares and Ridge regression  allow us to
derive quantities like the variance and other expectation values in a
rather straightforward way.


It is assumed that $\varepsilon_i
\sim \mathcal{N}(0, \sigma^2)$ and the $\varepsilon_{i}$ are
independent, i.e.: 
!bt
\begin{align*} 
\mbox{Cov}(\varepsilon_{i_1},
\varepsilon_{i_2}) & = \left\{ \begin{array}{lcc} \sigma^2 & \mbox{if}
& i_1 = i_2, \\ 0 & \mbox{if} & i_1 \not= i_2.  \end{array} \right.
\end{align*} 
!et
The randomness of $\varepsilon_i$ implies that
$\mathbf{y}_i$ is also a random variable. In particular,
$\mathbf{y}_i$ is normally distributed, because $\varepsilon_i \sim
\mathcal{N}(0, \sigma^2)$ and $\mathbf{X}_{i,\ast} \, \bm{\beta}$ is a
non-random scalar. To specify the parameters of the distribution of
$\mathbf{y}_i$ we need to calculate its first two moments. 

Recall that $\bm{X}$ is a matrix of dimensionality $n\times p$. The
notation above $\mathbf{X}_{i,\ast}$ means that we are looking at the
row number $i$ and perform a sum over all values $p$.


!split
===== Assumptions made =====

The assumption we have made here can be summarized as (and this is going to be useful when we discuss the bias-variance trade off)
that there exists a function $f(\bm{x})$ and  a normal distributed error $\bm{\varepsilon}\sim \mathcal{N}(0, \sigma^2)$
which describe our data
!bt
\[
\bm{y} = f(\bm{x})+\bm{\varepsilon}
\]
!et

We approximate this function with our model from the solution of the linear regression equations, that is our
function $f$ is approximated by $\bm{\tilde{y}}$ where we want to minimize $(\bm{y}-\bm{\tilde{y}})^2$, our MSE, with
!bt
\[
\bm{\tilde{y}} = \bm{X}\bm{\beta}.
\]
!et

!split
===== Expectation value and variance =====

We can calculate the expectation value of $\bm{y}$ for a given element $i$ 
!bt
\begin{align*} 
\mathbb{E}(y_i) & =
\mathbb{E}(\mathbf{X}_{i, \ast} \, \bm{\beta}) + \mathbb{E}(\varepsilon_i)
\, \, \, = \, \, \, \mathbf{X}_{i, \ast} \, \beta, 
\end{align*} 
!et
while
its variance is 
!bt
\begin{align*} \mbox{Var}(y_i) & = \mathbb{E} \{ [y_i
- \mathbb{E}(y_i)]^2 \} \, \, \, = \, \, \, \mathbb{E} ( y_i^2 ) -
[\mathbb{E}(y_i)]^2  \\  & = \mathbb{E} [ ( \mathbf{X}_{i, \ast} \,
\beta + \varepsilon_i )^2] - ( \mathbf{X}_{i, \ast} \, \bm{\beta})^2 \\ &
= \mathbb{E} [ ( \mathbf{X}_{i, \ast} \, \bm{\beta})^2 + 2 \varepsilon_i
\mathbf{X}_{i, \ast} \, \bm{\beta} + \varepsilon_i^2 ] - ( \mathbf{X}_{i,
\ast} \, \beta)^2 \\  & = ( \mathbf{X}_{i, \ast} \, \bm{\beta})^2 + 2
\mathbb{E}(\varepsilon_i) \mathbf{X}_{i, \ast} \, \bm{\beta} +
\mathbb{E}(\varepsilon_i^2 ) - ( \mathbf{X}_{i, \ast} \, \bm{\beta})^2 
\\ & = \mathbb{E}(\varepsilon_i^2 ) \, \, \, = \, \, \,
\mbox{Var}(\varepsilon_i) \, \, \, = \, \, \, \sigma^2.  
\end{align*}
!et
Hence, $y_i \sim \mathcal{N}( \mathbf{X}_{i, \ast} \, \bm{\beta}, \sigma^2)$, that is $\bm{y}$ follows a normal distribution with 
mean value $\bm{X}\bm{\beta}$ and variance $\sigma^2$ (not be confused with the singular values of the SVD). 

!split
===== Expectation value and variance for $\bm{\beta}$ =====

With the OLS expressions for the optimal parameters $\bm{\hat{\beta}}$ we can evaluate the expectation value
!bt
\[
\mathbb{E}(\bm{\hat{\beta}}) = \mathbb{E}[ (\mathbf{X}^{\top} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbb{E}[ \mathbf{Y}]=(\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T}\mathbf{X}\bm{\beta}=\bm{\beta}.
\]
!et
This means that the estimator of the regression parameters is unbiased.

We can also calculate the variance

The variance of the optimal value $\bm{\hat{\beta}}$ is
!bt
\begin{eqnarray*}
\mbox{Var}(\bm{\hat{\beta}}) & = & \mathbb{E} \{ [\bm{\beta} - \mathbb{E}(\bm{\beta})] [\bm{\beta} - \mathbb{E}(\bm{\beta})]^{T} \}
\\
& = & \mathbb{E} \{ [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} - \bm{\beta}] \, [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} - \bm{\beta}]^{T} \}
\\
% & = & \mathbb{E} \{ [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y}] \, [(\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y}]^{T} \} - \bm{\beta} \, \bm{\beta}^{T}
% \\
% & = & \mathbb{E} \{ (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \mathbf{Y} \, \mathbf{Y}^{T} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1}  \} - \bm{\beta} \, \bm{\beta}^{T}
% \\
& = & (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \, \mathbb{E} \{ \mathbf{Y} \, \mathbf{Y}^{T} \} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \bm{\beta} \, \bm{\beta}^{T}
\\
& = & (\mathbf{X}^{T} \mathbf{X})^{-1} \, \mathbf{X}^{T} \, \{ \mathbf{X} \, \bm{\beta} \, \bm{\beta}^{T} \,  \mathbf{X}^{T} + \sigma^2 \} \, \mathbf{X} \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \bm{\beta} \, \bm{\beta}^{T}
% \\
% & = & (\mathbf{X}^T \mathbf{X})^{-1} \, \mathbf{X}^T \, \mathbf{X} \, \bm{\beta} \, \bm{\beta}^T \,  \mathbf{X}^T \, \mathbf{X} \, (\mathbf{X}^T % \mathbf{X})^{-1}
% \\
% & & + \, \, \sigma^2 \, (\mathbf{X}^T \mathbf{X})^{-1} \, \mathbf{X}^T  \, \mathbf{X} \, (\mathbf{X}^T \mathbf{X})^{-1} - \bm{\beta} \bm{\beta}^T
\\
& = & \bm{\beta} \, \bm{\beta}^{T}  + \sigma^2 \, (\mathbf{X}^{T} \mathbf{X})^{-1} - \bm{\beta} \, \bm{\beta}^{T}
\, \, \, = \, \, \, \sigma^2 \, (\mathbf{X}^{T} \mathbf{X})^{-1},
\end{eqnarray*}
!et

where we have used  that $\mathbb{E} (\mathbf{Y} \mathbf{Y}^{T}) =
\mathbf{X} \, \bm{\beta} \, \bm{\beta}^{T} \, \mathbf{X}^{T} +
\sigma^2 \, \mathbf{I}_{nn}$. From $\mbox{Var}(\bm{\beta}) = \sigma^2
\, (\mathbf{X}^{T} \mathbf{X})^{-1}$, one obtains an estimate of the
variance of the estimate of the $j$-th regression coefficient:
$\bm{\sigma}^2 (\bm{\beta}_j ) = \bm{\sigma}^2 [(\mathbf{X}^{T} \mathbf{X})^{-1}]_{jj} $. This may be used to
construct a confidence interval for the estimates.


In a similar way, we can obtain analytical expressions for say the
expectation values of the parameters $\bm{\beta}$ and their variance
when we employ Ridge regression, allowing us again to define a confidence interval. 

It is rather straightforward to show that
!bt
\[
\mathbb{E} \big[ \bm{\beta}^{\mathrm{Ridge}} \big]=(\mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I}_{pp})^{-1} (\mathbf{X}^{\top} \mathbf{X})\bm{\beta}^{\mathrm{OLS}}.
\]
!et
We see clearly that 
$\mathbb{E} \big[ \bm{\beta}^{\mathrm{Ridge}} \big] \not= \bm{\beta}^{\mathrm{OLS}}$ for any $\lambda > 0$. We say then that the ridge estimator is biased.

We can also compute the variance as 

!bt
\[
\mbox{Var}[\bm{\beta}^{\mathrm{Ridge}}]=\sigma^2[  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}  \mathbf{X}^{T} \mathbf{X} \{ [  \mathbf{X}^{\top} \mathbf{X} + \lambda \mathbf{I} ]^{-1}\}^{T},
\]
!et
and it is easy to see that if the parameter $\lambda$ goes to infinity then the variance of Ridge parameters $\bm{\beta}$ goes to zero. 

With this, we can compute the difference 

!bt
\[
\mbox{Var}[\bm{\beta}^{\mathrm{OLS}}]-\mbox{Var}(\bm{\beta}^{\mathrm{Ridge}})=\sigma^2 [  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}[ 2\lambda\mathbf{I} + \lambda^2 (\mathbf{X}^{T} \mathbf{X})^{-1} ] \{ [  \mathbf{X}^{T} \mathbf{X} + \lambda \mathbf{I} ]^{-1}\}^{T}.
\]
!et
The difference is non-negative definite since each component of the
matrix product is non-negative definite. 
This means the variance we obtain with the standard OLS will always for $\lambda > 0$ be larger than the variance of $\bm{\beta}$ obtained with the Ridge estimator. This has interesting consequences when we discuss the so-called bias-variance trade-off below. 


!split
===== Deriving OLS from a probability distribution =====

Our basic assumption when we derived the OLS equations was to assume
that our output is determined by a given continuous function
$f(\bm{x})$ and a random noise $\bm{\epsilon}$ given by the normal
distribution with zero mean value and an undetermined variance
$\sigma^2$.

We found above that the outputs $\bm{y}$ have a mean value given by
$\bm{X}\hat{\bm{\beta}}$ and variance $\sigma^2$. Since the entries to
the design matrix are not stochastic variables, we can assume that the
probability distribution of our targets is also a normal distribution
but now with mean value $\bm{X}\hat{\bm{\beta}}$. This means that a
single output $y_i$ is given by the Gaussian distribution

!bt
\[
y_i\sim \mathcal{N}(\bm{X}_{i,*}\bm{\beta}, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}.
\]
!et

!split
===== Independent and Identically Distrubuted (iid) =====

We assume now that the various $y_i$ values are stochastically distributed according to the above Gaussian distribution. 
We define this distribution as
!bt
\[
p(y_i, \bm{X}\vert\bm{\beta})=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]},
\]
!et
which reads as finding the likelihood of an event $y_i$ with the input variables $\bm{X}$ given the parameters (to be determined) $\bm{\beta}$.

Since these events are assumed to be independent and identicall distributed we can build the probability distribution function (PDF) for all possible event $\bm{y}$ as the product of the single events, that is we have

!bt
\[
p(\bm{y},\bm{X}\vert\bm{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}=\prod_{i=0}^{n-1}p(y_i,\bm{X}\vert\bm{\beta}).
\]
!et

We will write this in a more compact form reserving $\bm{D}$ for the domain of events, including the ouputs (targets) and the inputs. That is
in case we have a simple one-dimensional input and output case
!bt
\[
\bm{D}=[(x_0,y_0), (x_1,y_1),\dots, (x_{n-1},y_{n-1})].
\]
!et
In the more general case the various inputs should be replaced by the possible features represented by the input data set $\bm{X}$. 
We can now rewrite the above probability as 
!bt
\[
p(\bm{D}\vert\bm{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}.
\]
!et

It is a conditional probability (see below) and reads as the likelihood of a domain of events $\bm{D}$ given a set of parameters $\bm{\beta}$.

!split
===== Maximum Likelihood Estimation (MLE) =====

In statistics, maximum likelihood estimation (MLE) is a method of
estimating the parameters of an assumed probability distribution,
given some observed data. This is achieved by maximizing a likelihood
function so that, under the assumed statistical model, the observed
data is the most probable. 


We will assume here that our events are given by the above Gaussian
distribution and we will determine the optimal parameters $\beta$ by
maximizing the above PDF. However, computing the derivatives of a
product function is cumbersome and can easily lead to overflow and/or
underflowproblems, with potentials for loss of numerical precision.


In practice, it is more convenient to maximize the logarithm of the
PDF because it is a monotonically increasing function of the argument.
Alternatively, and this will be our option, we will minimize the
negative of the logarithm since this is a monotonically decreasing
function.

Note also that maximization/minimization of the logarithm of the PDF
is equivalent to the maximization/minimization of the function itself.



!split
===== A new Cost Function =====

We could now define a new cost function to minimize, namely the negative logarithm of the above PDF

!bt
\[
C(\bm{\beta}=-\log{\prod_{i=0}^{n-1}p(y_i,\bm{X}\vert\bm{\beta})}=-\sum_{i=0}^{n-1}\log{p(y_i,\bm{X}\vert\bm{\beta})},
\]
!et
which becomes
!bt
\[
C(\bm{\beta}=\frac{n}{2}\log{2\pi\sigma^2}+\frac{\vert\vert (\bm{y}-\bm{X}\bm{\beta})\vert\vert_2^2}{2\sigma^2}.
\]
!et

Taking the derivative of the *new* cost function with respect to the parameters $\beta$ we recognize our familiar OLS equation, namely

!bt
\[
\bm{X}^T\left(\bm{y}-\bm{X}\bm{\beta}\right) =0,
\]
!et
which leads to the well-known OLS equation for the optimal paramters $\beta$
!bt
\[
\hat{\bm{\beta}}^{\mathrm{OLS}}=\left(\bm{X}^T\bm{X}\right)^{-1}\bm{X}^T\bm{y}!
\]
!et


Before we make a similar analysis for Ridge and Lasso regression, we need a short reminder on statistics. 

!split
===== More basic Statistics and Bayes' theorem =====

A central theorem in statistics is Bayes' theorem. This theorem plays a similar role as the good old Pythagoras' theorem in geometry.
Bayes' theorem is extremely simple to derive. But to do so we need some basic axioms from statistics.

Assume we have two domains of events $X=[x_0,x_1,\dots,x_{n-1}]$ and $Y=[y_0,y_1,\dots,y_{n-1}]$.

We define also the likelihood for $X$ and $Y$ as $p(X)$ and $p(Y)$ respectively.
The likelihood of a specific event $x_i$ (or $y_i$) is then written as $p(X=x_i)$ or just $p(x_i)=p_i$. 

!bblock Union of events is given by
!bt
\[
p(X \cup Y)= p(X)+p(Y)-p(X \cap Y).
\]
!et
!eblock


!bblock The product rule (aka joint probability) is given by
!bt
\[
p(X \cup Y)= p(X,Y)= p(X\vert Y)p(Y)=p(Y\vert X)p(X),
\]
!et
where we read $p(X\vert Y)$ as the likelihood of obtaining $X$ given $Y$.
!eblock

If we have independent events then $p(X,Y)=p(X)p(Y)$.


!split
===== Marginal Probability =====

The marginal probability is defined in terms of only one of the set of variables $X,Y$. For a discrete probability we have
!bblock 
!bt
\[
p(X)=\sum_{i=0}^{n-1}p(X,Y=y_i)=\sum_{i=0}^{n-1}p(X\vert Y=y_i)p(Y=y_i)=\sum_{i=0}^{n-1}p(X\vert y_i)p(y_i).
\]
!et
!eblock


!split
===== Conditional  Probability =====

The conditional  probability, if $p(Y) > 0$, is 
!bblock 
!bt
\[
p(X\vert Y)= \frac{p(X,Y)}{p(Y)}=\frac{p(X,Y)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)}.
\]
!et
!eblock


!split
===== Bayes' Theorem =====

If we combine the conditional probability with the marginal probability and the standard product rule, we have
!bt
\[
p(X\vert Y)= \frac{p(X,Y)}{p(Y)},
\]
!et
which we can rewrite as

!bt
\[
p(X\vert Y)= \frac{p(X,Y)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)}=\frac{p(Y\vert X)p(X)}{\sum_{i=0}^{n-1}p(Y\vert X=x_i)p(x_i)},
\]
!et
which is Bayes' theorem. It allows us to evaluate the uncertainty in in $X$ after we have observed $Y$. We can easily interchange $X$ with $Y$.  

!split
===== Interpretations of Bayes' Theorem =====

The quantity $p(Y\vert X)$ on the right-hand side of the theorem is
evaluated for the observed data $Y$ and can be viewed as a function of
the parameter space represented by $X$. This function is not
necesseraly normalized and is normally called the likelihood function.

The function $p(X)$ on the right hand side is called the prior while the function on the left hand side is the called the posterior probability. The denominator on the right hand side serves as a normalization factor for the posterior distribution.

Let us try to illustrate Bayes' theorem through an example.

!split
=====  Example of Usage of Bayes' theorem =====

Let us suppose that you are undergoing a series of mammography scans in
order to rule out possible breast cancer cases.  We define the
sensitivity for a positive event by the variable $X$. It takes binary
values with $X=1$ representing a positive event and $X=0$ being a
negative event. We reserve $Y$ as a classification parameter for
either a negative or a positive breast cancer confirmation. (Short note on wordings: positive here means having breast cancer, although none of us would consider this being a  positive thing).

We let $Y=1$ represent the the case of having breast cancer and $Y=0$ as not.

Let us assume that if you have breast cancer, the test will be positive with a probability of $0.8$, that is we have

!bt
\[
p(X=1\vert Y=1) =0.8.
\]
!et

This obviously sounds  scary since many would conclude that if the test is positive, there is a likelihood of $80\%$ for having cancer.
It is however not correct, as the following Bayesian analysis shows.

!split
===== Doing it correctly =====

If we look at various national surveys on breast cancer, the general likelihood of developing breast cancer is a very small number.
Let us assume that the prior probability in the population as a whole is

!bt
\[
p(Y=1) =0.004.
\]
!et

We need also to account for the fact that the test may produce a false positive result (false alarm). Let us here assume that we have
!bt
\[
p(X=1\vert Y=0) =0.1.
\]
!et

Using Bayes' theorem we can then find the posterior probability that the person has breast cancer in case of a positive test, that is we can compute

!bt
\[
p(Y=1\vert X=1)=\frac{p(X=1\vert Y=1)p(Y=1)}{p(X=1\vert Y=1)p(Y=1)+p(X=1\vert Y=0)p(Y=0)}=\frac{0.8\times 0.004}{0.8\times 0.004+0.1\times 0.996}=0.031.
\]
!et
That is, in case of a positive test, there is only a $3\%$ chance of having breast cancer!


!split
===== Bayes' Theorem and Ridge and Lasso Regression =====

Hitherto we have discussed Ridge and Lasso regression in terms of a
linear analysis. This may to many of you feel rather technical and
perhaps not that intuitive. The question is whether we can develop a
more intuitive way of understanding what Ridge and Lasso express.

Before we proceed let us perform a Ridge, Lasso  and OLS analysis of a polynomial fit. 

!split
===== Test Function for what happens with OLS, Ridge and Lasso =====

We will play around with a study of the values for the optimal
parameters $\bm{\beta}$ using OLS, Ridge and Lasso regression.  For
OLS, you will notice as function of the noise and polynomial degree,
that the parameters $\beta$ will fluctuate from order to order in the
polynomial fit and that for larger and larger polynomial degrees of freedom, the parameters will tend to increase in value for OLS.

For Ridge and Lasso regression, the higher order parameters will typically be reduced, providing thereby less fluctuations from one order to another one.

!bc pycod
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


!ec

How can we understand this?  


!split
===== Invoking Bayes' theorem =====

Using Bayes' theorem we can gain a better intuition about Ridge and Lasso regression. 

For ordinary least squares we postulated that the maximum likelihood for the doamin of events $\bm{D}$ (one-dimensional case)
!bt
\[
\bm{D}=[(x_0,y_0), (x_1,y_1),\dots, (x_{n-1},y_{n-1})],
\]
!et
is given by
!bt
\[
p(\bm{D}\vert\bm{\beta})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}.
\]
!et

In Bayes' theorem this function plays the role of the so-called likelihood. We could now ask the question what is the posterior probability of a parameter set $\bm{\beta}$ given a domain of events $\bm{D}$?  That is, how can we define the posterior probability 

!bt
\[
p(\bm{\beta}\vert\bm{D}).
\]
!et

Bayes' theorem comes to our rescue here since (omitting the normalization constant)
!bt
\[
p(\bm{\beta}\vert\bm{D})\propto p(\bm{D}\vert\bm{\beta})p(\bm{\beta}).
\]
!et

We have a model for $p(\bm{D}\vert\bm{\beta})$ but need one for the _prior_ $p(\bm{\beta}$!   


!split
===== Ridge and Bayes =====

With the posterior probability defined by a likelihood which we have
already modeled and an unknown prior, we are now ready to make
additional models for the prior.

We can, based on our discussions of the variance of $\bm{\beta}$ and the mean value, assume that the prior for the values $\bm{\beta}$ is given by a Gaussian with mean value zero and variance $\tau^2$, that is

!bt
\[
p(\bm{\beta})=\prod_{j=0}^{p-1}\exp{\left(-\frac{\beta_j^2}{2\tau^2}\right)}.
\]
!et

Our posterior probability becomes then (omitting the normalization factor which is just a constant)
!bt
\[
p(\bm{\beta\vert\bm{D})}=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}\prod_{j=0}^{p-1}\exp{\left(-\frac{\beta_j^2}{2\tau^2}\right)}.
\]
!et


We can now optimize this quantity with respect to $\bm{\beta}$. As we
did for OLS, this is most conveniently done by taking the negative
logarithm of the posterior probability. Doing so and leaving out the
constants terms that do not depend on $\beta$, we have


!bt
\[
C(\bm{\beta})=\frac{\vert\vert (\bm{y}-\bm{X}\bm{\beta})\vert\vert_2^2}{2\sigma^2}+\frac{1}{2\tau^2}\vert\vert\bm{\beta}\vert\vert_2^2,
\]
!et
and replacing $1/2\tau^2$ with $\lambda$ we have

!bt
\[
C(\bm{\beta})=\frac{\vert\vert (\bm{y}-\bm{X}\bm{\beta})\vert\vert_2^2}{2\sigma^2}+\lambda\vert\vert\bm{\beta}\vert\vert_2^2,
\]
!et
which is our Ridge cost function!  Nice, isn't it?

!split
===== Lasso and Bayes =====

To derive the Lasso cost function, we simply replace the Gaussian prior with an exponential distribution ("Laplace in this case":"https://en.wikipedia.org/wiki/Laplace_distribution") with zero mean value,  that is

!bt
\[
p(\bm{\beta})=\prod_{j=0}^{p-1}\exp{\left(-\frac{\vert\beta_j\vert}{\tau}\right)}.
\]
!et

Our posterior probability becomes then (omitting the normalization factor which is just a constant)
!bt
\[
p(\bm{\beta}\vert\bm{D})=\prod_{i=0}^{n-1}\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left[-\frac{(y_i-\bm{X}_{i,*}\bm{\beta})^2}{2\sigma^2}\right]}\prod_{j=0}^{p-1}\exp{\left(-\frac{\vert\beta_j\vert}{\tau}\right)}.
\]
!et


Taking the negative
logarithm of the posterior probability and leaving out the
constants terms that do not depend on $\beta$, we have


!bt
\[
C(\bm{\beta}=\frac{\vert\vert (\bm{y}-\bm{X}\bm{\beta})\vert\vert_2^2}{2\sigma^2}+\frac{1}{\tau}\vert\vert\bm{\beta}\vert\vert_1,
\]
!et
and replacing $1/\tau$ with $\lambda$ we have

!bt
\[
C(\bm{\beta}=\frac{\vert\vert (\bm{y}-\bm{X}\bm{\beta})\vert\vert_2^2}{2\sigma^2}+\lambda\vert\vert\bm{\beta}\vert\vert_1,
\]
!et
which is our Lasso cost function!  



===== Exercise: mean values and variances in linear regression  =====



This exercise deals with various mean values ad variances in  linear regression method (here it may be useful to look up chapter 3, equation (3.8) of "Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, The Elements of Statistical Learning, Springer":"https://www.springer.com/gp/book/9780387848570").

The assumption we have made is 
that there exists a function $f(\bm{x})$ and  a normal distributed error $\bm{\varepsilon}\sim \mathcal{N}(0, \sigma^2)$
which describes our data
!bt
\[
\bm{y} = f(\bm{x})+\bm{\varepsilon}
\]
!et

We then approximate this function with our model from the solution of the linear regression equations (ordinary least squares OLS), that is our
function $f$ is approximated by $\bm{\tilde{y}}$ where we minimized  $(\bm{y}-\bm{\tilde{y}})^2$, with
!bt
\[
\bm{\tilde{y}} = \bm{X}\bm{\beta}.
\]
!et
The matrix $\bm{X}$ is the so-called design matrix. 

!bsubex
Show that  the expectation value of $\bm{y}$ for a given element $i$ 
!bt
\begin{align*} 
\mathbb{E}(y_i) & =\mathbf{X}_{i, \ast} \, \beta, 
\end{align*} 
!et
and that
its variance is 
!bt
\begin{align*} \mbox{Var}(y_i) & = \sigma^2.  
\end{align*}
!et
Hence, $y_i \sim \mathcal{N}( \mathbf{X}_{i, \ast} \, \bm{\beta}, \sigma^2)$, that is $\bm{y}$ follows a normal distribution with 
mean value $\bm{X}\bm{\beta}$ and variance $\sigma^2$.

!esubex

!bsubex
With the OLS expressions for the parameters $\bm{\beta}$ show that
!bt
\[
\mathbb{E}(\bm{\beta}) = \bm{\beta}.
\]
!et
!esubex

!bsubex
Show finally that the variance of $\bm{\beta}$ is
!bt
\begin{eqnarray*}
\mbox{Var}(\bm{\beta}) & = & \sigma^2 \, (\mathbf{X}^{T} \mathbf{X})^{-1}.
\end{eqnarray*}
!et

!esubex


===== Exercise: Adding Ridge and Lasso Regression  =====


This exercise is a continuation of the exercises from week 35.

We will
use the same function to generate our data set, still staying with a
simple function $y(x)$ which we want to fit using linear regression,
but now extending the analysis to include the Ridge and the Lasso
regression methods. 

We will thus again generate our own dataset for a function $y(x)$ where 
$x \in [0,1]$ and defined by random numbers computed with the uniform
distribution. The function $y$ is a quadratic polynomial in $x$ with
added stochastic noise according to the normal distribution $\cal{N}(0,1)$.

The following simple Python instructions define our $x$ and $y$ values (with 100 data points).
!bc pycod
x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)
!ec

!bsubex
Write your own code for the Ridge method (see chapter 3.4 of Hastie *et al.*, equations (3.43) and (3.44)) and compute the parametrization for different values of $\lambda$. Study the dependence on $\lambda$ while also varying the strength of the noise in your expression for $y(x)$. 

!esubex

!bsubex
Our next step is to study the variance of the parameters $\beta_1$ and $\beta_2$ (assuming that we are parameterizing our function with a second-order polynomial). We will use standard linear regression and the Ridge regression.  You can now opt for either writing your own function or using _Scikit-Learn_ to find the parameters $\beta$. From your results calculate the variance of these parameters (recall that this is equal to the diagonal elements of the matrix $(\hat{X}^T\hat{X})+\lambda\hat{I})^{-1}$). Discuss the results of these variances as functions of $\lambda$. In particular, try to link your discussion with the discussion in Hastie *et al.* and their figures 3.10 and  3.11. _Scikit-Learn_ may not provide the variance of the parameters $\beta$. This needs to be checked. With your own code you can however do so.
!esubex






