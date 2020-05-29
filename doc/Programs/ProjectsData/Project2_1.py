# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:03 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# close all previous images
plt.close('all')

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ensure the same random numbers appear every time
np.random.seed(0)

# download breast cancer dataset
cancer = load_breast_cancer()

# define inputs and labels
inputs = cancer.data
outputs = cancer.target     #Malignant or bening
labels = cancer.feature_names[0:30]


print('The content of the breast cancer dataset is:')
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

n_inputs = len(inputs)

#%% VISUALIZATION

X = inputs
y = outputs

plt.figure()
plt.scatter(X[:,0], X[:,2], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius')
plt.ylabel('Mean perimeter')
plt.show()

plt.figure()
plt.scatter(X[:,5], X[:,6], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean compactness')
plt.ylabel('Mean concavity')
plt.show()


plt.figure()
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius')
plt.ylabel('Mean texture')
plt.show()

plt.figure()
plt.scatter(X[:,2], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter')
plt.ylabel('Mean compactness')
plt.show()


# %% COVARIANCE AND CORRELATION 

import pandas as pd
import seaborn as sns 

# Making a data frame

meanpd = pd.DataFrame(X[:,0:10],columns=labels[0:10])
corr = meanpd.corr().round(1)		# Compute pairwise correlation of columns, excluding NA/null values.

# use the heatmap function from seaborn to plot the correlation matrix
plt.figure()
sns.heatmap(corr, cbar = True,   annot=False,
           xticklabels= labels[0:10], yticklabels= labels[0:10],
           cmap= 'YlOrRd')


X_t = X[ : , 1:3]

clf = LogisticRegression()
clf.fit(X_t, y)


# Set min and max values and give it some padding
x_min, x_max = X_t[:, 1].min() - .5, X_t[:, 1].max() + .5
y_min, y_max = X_t[:, 0].min() - .5, X_t[:, 0].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 2], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter')
plt.ylabel('Mean texture')
plt.title('Logistic Regression')
plt.show()

# %% TRAIN AND TEST DATASET

# Set up training data: from scikit-learn library
train_size = 0.9
test_size = 1 - train_size

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, train_size=train_size,
                                                    test_size=test_size)

# %% LOGISTIC REGRESSION and ACCURACY

print('----------------------')
print('LOGISTIC REGRESSION')
print('----------------------')
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Train set accuracy with Logistic Regression:: {:.2f}".format(logreg.score(X_train,y_train)))
print("Test set accuracy with Logistic Regression:: {:.2f}".format(logreg.score(X_test,y_test)))

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, y_train)
print("Train set accuracy Logistic Regression scaled data: {:.2f}".format(logreg.score(X_train_scaled,y_train)))
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))


# %% CROSS VALIDATION FROM SCIKIT-LEARN
from sklearn.linear_model import LogisticRegressionCV

print('----------------------')
print('LOGISTIC REGRESSION with CROSS VALIDATION 5-KFold')
print('----------------------')
logreg = LogisticRegressionCV()
logreg.fit(X_train, y_train)
print("Train set accuracy with Logistic Regression, CV:: {:.2f}".format(logreg.score(X_train,y_train)))
print("Test set accuracy with Logistic Regression, CV:: {:.2f}".format(logreg.score(X_test,y_test)))

# Scale data

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, y_train)
print("Train set accuracy Logistic Regression scaled data: {:.2f}".format(logreg.score(X_train_scaled,y_train)))
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))

# %% CROSS VALIDATION: OUR OWN CODE 

"""Implement cross-validation framework (only on liblinear solver)"""

#Initiate k-fold instance for implementing manual cross-validation using KFold

import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LogReg
import os 

"""Generate training and testing datasets"""

x=inputs

#Select features relevant to classification (texture,perimeter,compactness and symmetery) 
#and add to input matrix

temp1=np.reshape(inputs[:,1],(len(inputs[:,1]),1))
temp2=np.reshape(inputs[:,2],(len(inputs[:,2]),1))
X=np.hstack((temp1,temp2))      
temp=np.reshape(inputs[:,5],(len(inputs[:,5]),1))
X=np.hstack((X,temp))       
temp=np.reshape(inputs[:,8],(len(inputs[:,8]),1))
X=np.hstack((X,temp))       

lamda=np.logspace(-5,5,11)    #Define array of hyperparameters


"""Implement K-fold cross-validation"""

k=5
kfold=KFold(n_splits=k)

train_scores=np.zeros((len(lamda),k))
test_scores=np.zeros((len(lamda),k))

for i in range(len(lamda)):
    j=0
    for train_inds,test_inds in kfold.split(X):
        X_train=X[train_inds]
        y_train=y[train_inds]
        X_test=X[test_inds]
        y_test=y[test_inds]
        
        
        clf=LogReg(C=1.0/lamda[i],random_state=1,verbose=0,max_iter=1E+3,tol=1E-5,solver='liblinear')
        clf.fit(X_train,y_train)
        train_scores[i,j]=clf.score(X_train,y_train)
        test_scores[i,j]=clf.score(X_test,y_test)
        j+=1
        
train_accuracy_cv_kfold=np.mean(train_scores,axis=1)
test_accuracy_cv_kfold=np.mean(test_scores,axis=1)

    
"""Plot results after K-fold cross validation"""

plt.figure()
plt.semilogx(lamda,train_accuracy_cv_kfold,'*-b',label='Training')
plt.semilogx(lamda,test_accuracy_cv_kfold,'*-r',label='Test')

plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('Accuracy')
plt.title('Accuracy LogReg (5 k-Fold CV)')
plt.show()


# %% DECISION TREES: CLASSIFICATION and ACCURACY


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Create the encoder.
encoder = OneHotEncoder(handle_unknown="ignore")
# Assume for simplicity all features are categorical.
encoder.fit(X)    
# Apply the encoder.
X = encoder.transform(X)


# Classification tree: with and without scaling (sc)

DEPTH=np.arange(start=1,stop=11,step=1)
test_acc = np.zeros(len(DEPTH))
test_acc_sc = np.zeros(len(DEPTH))

for i in DEPTH:
    tree_clf = DecisionTreeClassifier(max_depth= i)
    tree_clf.fit(X_train, y_train)   
    test_acc[i-1] = tree_clf.score(X_test,y_test)   
    print("Decision Tree (No Max depth): {:.2f}".format(DEPTH[i-1]))
    print("               Test accuracy: {:.2f}".format(test_acc[i-1]))
    
    export_graphviz(
            tree_clf,
            out_file="ride.dot",
            rounded=True,
            filled=True
    )
    cmd = 'dot -Tpng ride.dot -o DecisionTree_max_depth_{:.2f}.png'.format(DEPTH[i-1])
    os.system(cmd)
    
#PLOT TEST ACCURACY
fig,p1=plt.subplots() 
p1.plot(DEPTH, test_acc, label='Test accuracy')


p1.set_xlabel('Max_depth in Decision Tree')
p1.set_ylabel('Accuracy')
p1.set_title("Decision Tree Test Accuracy", fontsize=18)
p1.legend()
 

tree_clf = DecisionTreeClassifier(max_depth=None)
tree_clf.fit(X_train, y_train)
print("Test set accuracy with Decision Tree (No Max depth): {:.2f}".format(tree_clf.score(X_test,y_test)))


# %% RANDOM FOREST and ACCURACY
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

print('RANDOM FOREST')

model=RandomForestClassifier(n_estimators= 100)# a simple random forest model
model.fit(X_train,y_train)# now fit our model for training data
y_pred = model.predict(X_test)# predict for the test data
RFtest_acc = metrics.accuracy_score(y_pred,y_test) # to check the accuracy


print("Test set accuracy with RANDOM FOREST: {:.2f}".format(RFtest_acc))
