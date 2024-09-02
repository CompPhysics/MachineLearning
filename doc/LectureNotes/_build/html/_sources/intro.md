# Applied Data Analysis and Machine Learning

## Introduction

Probability theory and statistical methods play a central role in Science. Nowadays we are
surrounded by huge amounts of data. For example, there are more than one trillion web pages; more than one
hour of video is uploaded to YouTube every second, amounting to years of content every
day; the genomes of 1000s of people, each of which has a length of more than a billion  base pairs, have
been sequenced by various labs and so on. This deluge of data calls for automated methods of data analysis,
which is exactly what machine learning aims at providing. 

## Learning outcomes

This course aims at giving you insights and knowledge about  many of the central algorithms used in Data Analysis and Machine Learning.  The course is project based and through  various numerical projects and weekly exercises you will be exposed to fundamental research problems in these fields, with the aim to reproduce state of the art scientific results. Both supervised and unsupervised methods will be covered. The emphasis is on a frequentist approach with an emphasis on predictions and correaltions. However, we will try, where appropriate, to link our machine learning models with a Bayesian approach as well. You will learn to develop and structure large codes for studying different cases where Machine Learning is applied to, get acquainted with computing facilities and learn to handle large scientific projects. A good scientific and ethical conduct is emphasized throughout the course. More specifically, after this course you will

- Learn about basic data analysis, statistical analysis, Bayesian statistics, Monte Carlo sampling, data optimization and machine learning;
- Be capable of extending the acquired knowledge to other systems and cases;
- Have an understanding of central algorithms used in data analysis and machine learning;
- Understand linear methods for regression and classification, from ordinary least squares, via Lasso and Ridge to Logistic regression and Kernel regression;
- Learn about neural networks and deep  learning methods for supervised and unsupervised learning. Emphasis on feed forward neural networks, convolutional and recurrent neural networks; 
- Learn about about decision trees, random forests, bagging and boosting methods;
- Learn about support vector machines and kernel transformations;
- Reduction of data sets and unsupervised learning, from PCA to clustering;
- Autoencoders and Reinforcement Learning;
- Work on numerical projects to illustrate the theory. The projects play a central role and you are expected to know modern programming languages like Python or C++ and/or Fortran (Fortran2003 or later).  

## Prerequisites and background

Basic knowledge in programming and mathematics, with an emphasis on linear algebra. Knowledge of Python or/and C++ as programming languages is strongly recommended and experience with Jupyter notebooks is recommended. Required courses are the equivalents to the University of Oslo mathematics courses MAT1100, MAT1110, MAT1120 and at least one of the corresponding computing and programming courses INF1000/INF1110 or MAT-INF1100/MAT-INF1100L/BIOS1100/KJM-INF1100. Most universities offer nowadays a basic programming course (often compulsory) where Python is the recurring programming language.
We recommend also refreshing your knowledge on Statistics and Probability theory. The lecture notes at https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/intro.html offer a review of Statistics and Probability theory.

## The course has two central parts

1. Statistical analysis and optimization of data
2. Machine learning


### Statistical analysis and optimization of data

The following topics will be covered
- Basic concepts, expectation values, variance, covariance, correlation functions and errors;
- Simpler models, binomial distribution, the Poisson distribution, simple and multivariate normal distributions;
- Central elements of Bayesian statistics and modeling;
- Gradient methods for data optimization, 
- Monte Carlo methods, Markov chains, Gibbs sampling and Metropolis-Hastings sampling;
- Estimation of errors and resampling techniques such as the cross-validation, blocking, bootstrapping and jackknife methods;
- Principal Component Analysis (PCA) and its mathematical foundation

### Machine learning

The following topics will be covered:
- Linear Regression and Logistic Regression;
- Neural networks and deep learning, including convolutional and recurrent neural networks
- Decisions trees, Random Forests, Bagging and Boosting
- Support vector machines
- Bayesian linear and logistic regression
- Boltzmann Machines
- Unsupervised learning Dimensionality reduction, PCA, k-means and  clustering
- Autoenconders

Hands-on demonstrations, exercises and projects aim at deepening your understanding of these topics.

Computational aspects play a central role and you are
expected to work on numerical examples and projects which illustrate
the theory and various algorithms discussed during the lectures. We recommend strongly to form small project groups of 2-3 participants, if possible. 



## Required Technologies

Course participants are expected to have their own laptops/PCs. We use _Git_ as version control software and the usage of providers like _GitHub_, _GitLab_ or similar are strongly recommended. If you are not familiar with Git as version control software, the following video may be of interest, see https://www.youtube.com/watch?v=RGOj5yH7evk&ab_channel=freeCodeCamp.org

We will make extensive use of Python as programming language and its
myriad of available libraries.  You will find
Jupyter notebooks invaluable in your work.  You can run _R_
codes in the Jupyter/IPython notebooks, with the immediate benefit of
visualizing your data. You can also use compiled languages like C++,
Rust, Julia, Fortran etc if you prefer. The focus in these lectures will be 
on Python.


If you have Python installed and you feel
pretty familiar with installing different packages, we recommend that
you install the following Python packages via _pip_ as 

* pip install numpy scipy matplotlib ipython scikit-learn mglearn sympy pandas pillow 

For OSX users we recommend, after having installed Xcode, to
install _brew_. Brew allows for a seamless installation of additional
software via for example 

* brew install python3

For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution,
you can use _pip_ as well and simply install Python as 

* sudo apt-get install python3

### Python installers

If you don't want to perform these operations separately and venture
into the hassle of exploring how to set up dependencies and paths, we
recommend two widely used distrubutions which set up all relevant
dependencies for Python, namely 

* Anaconda:https://docs.anaconda.com/, 

which is an open source
distribution of the Python and R programming languages for large-scale
data processing, predictive analytics, and scientific computing, that
aims to simplify package management and deployment. Package versions
are managed by the package management system _conda_. 

* Enthought canopy:https://www.enthought.com/product/canopy/ 

is a Python
distribution for scientific and analytic computing distribution and
analysis environment, available for free and under a commercial
license.

Furthermore, Google's Colab:https://colab.research.google.com/notebooks/welcome.ipynb is a free Jupyter notebook environment that requires 
no setup and runs entirely in the cloud. Try it out!

### Useful Python libraries
Here we list several useful Python libraries we strongly recommend (if you use anaconda many of these are already there)

* _NumPy_:https://www.numpy.org/ is a highly popular library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* _The pandas_:https://pandas.pydata.org/ library provides high-performance, easy-to-use data structures and data analysis tools 
* _Xarray_:http://xarray.pydata.org/en/stable/ is a Python package that makes working with labelled multi-dimensional arrays simple, efficient, and fun!
* _Scipy_:https://www.scipy.org/ (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. 
* _Matplotlib_:https://matplotlib.org/ is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* _Autograd_:https://github.com/HIPS/autograd can automatically differentiate native Python and Numpy code. It can handle a large subset of Python's features, including loops, ifs, recursion and closures, and it can even take derivatives of derivatives of derivatives
* _JAX_ https://jax.readthedocs.io/en/latest/index.html has now more or less replaced _Autograd_.
  JAX is Autograd and XLA, brought together for high-performance numerical computing and machine learning research.
  It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU, and more.
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 



 
