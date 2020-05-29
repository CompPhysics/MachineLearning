import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

# Reading data using PANDA 
data = pd.read_csv("pulsar_stars.csv")
data.head()
#DATA
targets = data["target_class"]
features = data.drop("target_class", axis = 1)
np.random.seed(2018)
#Split data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 66)


# Define the learning rate, hyperparameter using NUMPY 
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
n_hidden_neurons = 50
epochs = 100

# Use scikit learn for neural network 
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver='adam')
        dnn.fit(X_train, y_train)
        DNN_scikit[i][j] = dnn
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, y_test))
        print()

        
#Plot the accuracy as function of learning rate and hyperparameter        
sns.set() 
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        train_pred = dnn.predict(X_train) 
        test_pred = dnn.predict(X_test)
        train_accuracy[i][j] = accuracy_score(y_train, train_pred)
        test_accuracy[i][j] = accuracy_score(y_test, test_pred)
        
fig, ax = plt.subplots(figsize = (10, 10))        
sns.heatmap(train_accuracy, annot=True,annot_kws={"size": 18}, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy",fontsize=18)
ax.set_ylabel("$\eta$",fontsize=18)
ax.set_yticklabels(eta_vals)
ax.set_xlabel("$\lambda$",fontsize=18)
ax.set_xticklabels(lmbd_vals)
plt.tick_params(labelsize=18)
 
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True,annot_kws={"size": 18}, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy",fontsize=18)
ax.set_ylabel("$\eta$",fontsize=18)
ax.set_yticklabels(eta_vals)
ax.set_xlabel("$\lambda$",fontsize=18)
ax.set_xticklabels(lmbd_vals)
plt.tick_params(labelsize=18)
#plt.show()        

#Plot confusion matrix at optimal values of learning rate and hyperameter
dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',

                            alpha=0.001, learning_rate_init=0.001, max_iter=epochs, solver='adam')
dnn.fit(X_train,y_train)
y_pred=dnn.predict(X_test)
fig1, ax = plt.subplots(figsize = (13,10))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt = "d",linecolor="k",linewidths=3)
ax.set_xlabel('True label',fontsize=18)
ax.set_ylabel('Predicted label',fontsize=18)
ax.set_title("CONFUSION MATRIX",fontsize=20)
plt.tick_params(labelsize=18)
plt.show()

# Feature importance -->weights
coef=dnn.coefs_[0]
print (coef)

