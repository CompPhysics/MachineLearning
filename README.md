# Applied Data Analysis and Machine Learning


This site contains all material relevant for the course on Applied
Data Analysis and Machine Learning.

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

Basic knowledge in programming and mathematics, with an emphasis on
linear algebra. Knowledge of Python or/and C++ as programming
languages is strongly recommended and experience with Jupyter
notebooks is recommended.
We recommend also refreshing your knowledge on Statistics and Probability
theory. The lecture notes at
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/intro.html
offer a review of Statistics and Probability theory.

## The course has two central parts

1. Statistical analysis and optimization of data
2. Machine learning algorithms and Deep Learning


### Statistical analysis and optimization of data

The following topics are normally be covered
- Basic concepts, expectation values, variance, covariance, correlation functions and errors;
- Simpler models, binomial distribution, the Poisson distribution, simple and multivariate normal distributions;
- Central elements of Bayesian statistics and modeling;
- Gradient methods for data optimization, 
- Monte Carlo methods, Markov chains, Gibbs sampling and Metropolis-Hastings sampling;
- Estimation of errors and resampling techniques such as the cross-validation, blocking, bootstrapping and jackknife methods;
- Principal Component Analysis (PCA) and its mathematical foundation

### Machine learning

The following topics are typically  covered:
- Linear Regression and Logistic Regression;
- Neural networks and deep learning, including convolutional and recurrent neural networks
- Decisions trees, Random Forests, Bagging and Boosting
- Support vector machines
- Bayesian linear and logistic regression
- Boltzmann Machines and generative models
- Unsupervised learning Dimensionality reduction, PCA, k-means and  clustering
- Autoenconders

Hands-on demonstrations, exercises and projects aim at deepening your understanding of these topics.

Computational aspects play a central role and you are
expected to work on numerical examples and projects which illustrate
the theory and various algorithms discussed during the lectures. We recommend strongly to form small project groups of 2-3 participants, if possible. 

## Instructor information
* _Name_: Morten Hjorth-Jensen
* _Email_: morten.hjorth-jensen@fys.uio.no
* _Phone_: +47-48257387
* _Office_: Department of Physics, University of Oslo, Eastern wing, room FØ470 
* _Office hours_: *Anytime*! Individual or group office hours can be performed either in person or via zoom. Feel free to send an email for planning. 


##  Teaching Assistants FS22
* Karl Henrik Fredly, k.h.fredly@fys.uio.no
* Fahimeh Najafi, fahimeh.najafi@fys.uio.no
* Ida Torkjellsdatter Storehaug, i.t.storehaug@fys.uio.no
* Mia-Katrin Ose Kvalsund, m.k.o.kvalsund@fys.uio.no
* Daniel Haas Becattini Lima, d.h.b.lima@fys.uio.no
* Adam Jakobsen, adam.jakobsen@fys.uio.no




## Practicalities

1. Four lectures per week, Fall semester, 10 ECTS. The lectures will be recorded and linked to this site and the official University of Oslo website for the course;
2. Two hours of laboratory sessions for work on computational projects and exercises for each group. There will  also be fully digital laboratory sessions for those who cannot attend;
3. Three projects which are graded and count 1/3 each of the final grade;
4. A selected number of weekly assignments;
5. The course is part of the CS Master of Science program, but is open to other bachelor and Master of Science students at the University of Oslo;
6. The course is offered as a so-called _cloned_ course,  FYS-STK4155 at the Master of Science level and FYS-STK3155 as a senior undergraduate)course;
7. Videos of teaching material are available via the links at https://compphysics.github.io/MachineLearning/doc/web/course.html;
8. Weekly email with summary of activities will be mailed to all participants;


## Grading
Grading scale: Grades are awarded on a scale from A to F, where A is the best grade and F is a fail. There are three projects which are graded and each project counts 1/3 of the final grade. The total score is thus the average from all three projects.

The final number of points is based on the average of all projects (including eventual additional points) and the grade follows the following table:

 * 92-100 points: A
 * 77-91 points: B
 * 58-76 points: C
 * 46-57 points: D
 * 40-45 points: E
 * 0-39 points: F-failed

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
* _JAX_" https://jax.readthedocs.io/en/latest/index.html has now more or less replaced _Autograd_.
  JAX is Autograd and XLA, brought together for high-performance numerical computing and machine learning research.
  It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU, and more.
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics.

* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 

## Textbooks

_Recommended textbooks_:
The lecture notes are collected as a jupyter-book at https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/intro.html. In addition to the electure notes, we recommend the books of Bishop and Goodfellow et al. We will follow these texts closely and the weekly reading assignments refer to these two texts.
- Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, https://www.springer.com/gp/book/9780387310732. This is the main textbook and this course covers chapters 1-7, 11 and 12. You can download for free the textbook in PDF format at https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. The different chapters are available for free at https://www.deeplearningbook.org/. Chapters 2-14 are highly recommended. The lectures follow to a good extent this text.
The weekly plans will include reading suggestions from these two textbooks. In addition, you may find the following textbooks interesting.
_Additional textbooks_:
- Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, The Elements of Statistical Learning, Springer, https://www.springer.com/gp/book/9780387848570. This is a well-known text and serves as additional literature.
- Aurelien Geron, Hands‑On Machine Learning with Scikit‑Learn and TensorFlow, O'Reilly, https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/. This text is very useful since it contains many code examples and hands-on applications of all algorithms discussed in this course.




_General learning book on statistical analysis_:
- Christian Robert and George Casella, Monte Carlo Statistical Methods, Springer
- Peter Hoff, A first course in Bayesian statistical models, Springer

_General Machine Learning Books_:
- Kevin Murphy, Machine Learning: A Probabilistic Perspective, MIT Press
- David J.C. MacKay, Information Theory, Inference, and Learning Algorithms, Cambridge University Press
- David Barber, Bayesian Reasoning and Machine Learning, Cambridge University Press 

## Links to relevant courses at the University of Oslo
The link here https://www.mn.uio.no/english/research/about/centre-focus/innovation/data-science/studies/  gives an excellent overview of courses on Machine learning at UiO.

- _STK2100 Machine learning and statistical methods for prediction and classification_ http://www.uio.no/studier/emner/matnat/math/STK2100/index-eng.html. 
- _IN3050 Introduction to Artificial Intelligence and Machine Learning_ https://www.uio.no/studier/emner/matnat/ifi/IN3050/index-eng.html. Introductory course in machine learning and AI with an algorithmic approach. 
- _STK-INF3000/4000 Selected Topics in Data Science_ http://www.uio.no/studier/emner/matnat/math/STK-INF3000/index-eng.html. The course provides insight into selected contemporary relevant topics within Data Science. 
- _IN4080 Natural Language Processing_ https://www.uio.no/studier/emner/matnat/ifi/IN4080/index.html. Probabilistic and machine learning techniques applied to natural language processing. 
- _STK-IN4300 Statistical learning methods in Data Science_ https://www.uio.no/studier/emner/matnat/math/STK-IN4300/index-eng.html. An advanced introduction to statistical and machine learning. For students with a good mathematics and statistics background.
- _INF4490 Biologically Inspired Computing_ http://www.uio.no/studier/emner/matnat/ifi/INF4490/. An introduction to self-adapting methods also called artificial intelligence or machine learning. 
- _IN-STK5000  Adaptive Methods for Data-Based Decision Making_ https://www.uio.no/studier/emner/matnat/ifi/IN-STK5000/index-eng.html. Methods for adaptive collection and processing of data based on machine learning techniques. 
- _IN5400/INF5860 Machine Learning for Image Analysis_ https://www.uio.no/studier/emner/matnat/ifi/IN5400/. An introduction to deep learning with particular emphasis on applications within Image analysis, but useful for other application areas too.
- _TEK5040 Deep learning for autonomous systems_ https://www.uio.no/studier/emner/matnat/its/TEK5040/. The course addresses advanced algorithms and architectures for deep learning with neural networks. The course provides an introduction to how deep-learning techniques can be used in the construction of key parts of advanced autonomous systems that exist in physical environments and cyber environments.
- _STK4051 Computational Statistics_ https://www.uio.no/studier/emner/matnat/math/STK4051/index-eng.html
- _STK4021 Applied Bayesian Analysis and Numerical Methods_ https://www.uio.no/studier/emner/matnat/math/STK4021/






 

