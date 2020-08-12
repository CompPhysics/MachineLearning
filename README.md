# FYS-STK3155/4155 Applied Data Analysis and Machine Learning, http://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html


This site contains all material relevant for the course on Applied Data Analysis and Machine Learning FYS-STK3155/4155 at the University of Oslo, Norway.

## Introduction

Probability theory and statistical methods play a central role in science. Nowadays we are
surrounded by huge amounts of data. For example, there are about one trillion web pages; more than one
hour of video is uploaded to YouTube every second, amounting to years of content every
day; the genomes of 1000s of people, each of which has a length of more than a billion  base pairs, have
been sequenced by various labs and so on. This deluge of data calls for automated methods of data analysis,
which is exactly what machine learning aims at providing. 

## Learning outcomes

This course aims at giving you insights and knowledge about  many of the central algorithms used in Data Analysis and Machine Learning.  The course is project based and through  various numerical projects, normally three, you will be exposed to fundamental research problems in these fields, with the aim to reproduce state of the art scientific results. Both supervised and unsupervised methods will be covered. The emphasis is on a frequentist approach, although we will try to link it with a Bayesian approach as well. You will learn to develop and structure large codes for studying different cases where Machine Learning is applied to, get acquainted with computing facilities and learn to handle large scientific projects. A good scientific and ethical conduct is emphasized throughout the course. More specifically, after this course you will

- Learn about basic data analysis, statistical analysis, Bayesian statistics, Monte Carlo sampling, data optimization and machine learning;
- Be capable of extending the acquired knowledge to other systems and cases;
- Have an understanding of central algorithms used in data analysis and machine learning;
- Understand linear methods for regression and classification, from ordinary least squares, via Lasso and Ridge to Logistic regression;
- Learn about neural networks and deep  learning methods for supervised and unsupervised learning. Emphasis on feed forward neural networks, convolutional and recurrent neural networks; 
- Learn about about decision trees, random forests, bagging and boosting methods;
- Learn about support vector machines and kernel transformations;
- Reduction of data sets, from PCA to clustering;
- Autoencoders and Reinforcement Learning;
- Work on numerical projects to illustrate the theory. The projects play a central role and you are expected to know modern programming languages like Python or C++ and/or Fortran (Fortran2003 or later).  

## Prerequisites

Basic knowledge in programming and mathematics, with an emphasis on linear algebra. Knowledge of Python or/and C++ as programming languages is strongly recommended and experience with Jupiter notebook is recommended. Required courses are the equivalents to the University of Oslo mathematics courses MAT1100, MAT1110, MAT1120 and at least one of the corresponding computing and programming courses INF1000/INF1110 or MAT-INF1100/MAT-INF1100L/BIOS1100/KJM-INF1100. Most universities offer nowadays a basic programming course (often compulsory) where Python is the recurring programming language.


## The course has two central parts

1. Statistical analysis and optimization of data
2. Machine learning

These topics will be scattered thorughout the course and may not  necessarily be taught separately. Rather, we will often take an approach (during the lectures and project/exercise sessions) where say elements from statistical data analysis are mixed with specific Machine Learning algorithms. 

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
- Unsupervised learning Dimensionality reduction, from PCA to cluster models

Hands-on demonstrations, exercises and projects aim at deepening your understanding of these topics.

Computational aspects play a central role and you are
expected to work on numerical examples and projects which illustrate
the theory and varous algorithms discussed during the lectures. We recommend strongly to form small project groups of 2-3 participants, if possible. 

## Instructor information
* _Name_: Morten Hjorth-Jensen
* _Email_: morten.hjorth-jensen@fys.uio.no
* _Phone_: +47-48257387
* _Office_: Department of Physics, University of Oslo, Eastern wing, room FØ470 
* _Office hours_: *Anytime*! In Fall Semester 2020 (FS20), as a rule of thumb office hours are planned via computer or telephone. Individual or group office hours will be performed via zoom. Feel free to send an email for planning. In person meetings may also be possible if allowed by the University of Oslo's COVID-19 instructions (see below for links).


##  Teaching Assistants FS20
* Øyvind Sigmundson Schøyen, oyvinssc@student.matnat.uio.no 	 
* Michael Bitney, m.s.bitney@fys.uio.no
* Kristian Wold, kriswold@student.matnat.uio.no
* Nicolai Haug, nicoha@student.matnat.uio.no
* Per-Dimitri Sønsteland, perdimitri.bs@gmail.com

## Practicalities
This course will be delivered in a hybrid mode, with online lectures and on site or online laboratory sessions. 

1. Four lectures per week, Fall semester, 10 ECTS. The lectures will be fully online. The lectures will be recorded and linked to this site and the official University of Oslo website for the course;
2. Two hours of laboratory sessions for work on computational projects and exercises for each group. Due to social distancing, at most 15 participants can attend. There will  also be fully digital laboratory sessions for those who cannot attend;
3. Three projects which are graded and count 1/3 each of the final grade;
4. A selected number of weekly assignments;
5. The course is part of the CS Master of Science program, but is open to other bachelor and Master of Science students at the University of Oslo;
6. The course is offered as a FYS-MAT4155 (Master of Science level) and a FYS-MAT3155 (senior undergraduate) course;
7. We use Piazza for course communication, a special link on how to register to Piazza can be found at the official University of Oslo page for the course or just use the link here https://piazza.com/uio.no/fall2020/fysstk4155. Slack is also used for course communication. The Slack link is https://machinelearninguio.slack.com ;
8. Videos of teaching material are available via the links at https://compphysics.github.io/MachineLearning/doc/web/course.html;
9. Weekly emails with summary of activities will be mailed to all participants;

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

Course participants are expected to have their own laptops/PCs. We use _Git_ as version control software and the usage of providers like _GitHub_, _GitLab_ or similar are strongly recommended.

We will make extensive use of Python as programming language and its
myriad of available libraries.  You will find
Jupyter notebooks invaluable in your work.  You can run _R_
codes in the Jupyter/IPython notebooks, with the immediate benefit of
visualizing your data. You can also use compiled languages like C++,
Rust, Julia, Fortran etc if you prefer. The focus in these lectures will be
on Python.


If you have Python installed (we strongly recommend Python3) and you feel
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
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 

## Possible textbooks

_Recommended textbooks_:
- Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, The Elements of Statistical Learning, Springer
- Aurelien Geron, Hands‑On Machine Learning with Scikit‑Learn and TensorFlow, O'Reilly

_General learning book on statistical analysis_:
- Christian Robert and George Casella, Monte Carlo Statistical Methods, Springer
- Peter Hoff, A first course in Bayesian statistical models, Springer

_General Machine Learning Books_:
- Kevin Murphy, Machine Learning: A Probabilistic Perspective, MIT Press
- Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer
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



## Personal Hygiene
All participants attending the laboratory sessions must maintain proper hygiene and health practices, including:
* frequently wash with soap and water or, if soap is unavailable, using hand sanitizer with at least 60% alcohol;
* Routinely cleaning and sanitizing living spaces and/or workspace;
* Using the bend of the elbow or shoulder to shield a cough or sneeze;
* Refraining from shaking hands;

## Adherence to Signage and Instructions 
Course participants  will (a) look for instructional signs posted by UiO or public health authorities, (b) observe instructions from UiO or public health authorities that are emailed to my “uio.no” account, and (c) follow those instructions.
The relevant links are https://www.uio.no/om/hms/korona/index.html and https://www.uio.no/om/hms/korona/retningslinjer/veileder-smittevern.html

## Self-Monitoring
Students will self-monitor for flu-like symptoms (for example, cough, shortness of breath, difficulty breathing, fever, sore throat or loss of taste or smell). If a student experiences any flu-like symptoms, they will stay home and contact a health care provider to determine what steps should be taken.
## Exposure to COVID-19 
If a student is exposed to someone who is ill or has tested positive for the COVID-19 virus, they will stay home, contact a health care provider and follow all public health recommendations. You may also contact the study administration of the department where you are registered as student. 
## Compliance and reporting 
Those who come to UiO facilities must commit to the personal responsibility necessary for us to remain as safe as possible, including following the specific guidelines outlined in this syllabus and provided by UiO more broadly (see links below here). 

## Additional information
See https://www.uio.no/om/hms/korona/index.html and https://www.uio.no/om/hms/korona/retningslinjer/veileder-smittevern.html. For English version, click on the relevant link.
