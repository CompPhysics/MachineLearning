# # Finding eigenvalues of matrices with neural networks. 
# Script for finding the eigenvectors corresponding to the largest eigenvalue of a matrix with a neural network.

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
# tf.set_random_seed(343)

# import tensorflow as tf
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#from lib import compute_dx_dt

matrix_size = 6

A = np.random.random_sample(size=(matrix_size,matrix_size))
A = (A.T + A)/2.0
start_matrix = A

eigen_vals, eigen_vecs =  np.linalg.eig(A)

A = tf.convert_to_tensor(A)
print("A = ", A)

x_0 = tf.convert_to_tensor(np.random.random_sample(size = (1,matrix_size)))
print("x0 = ", x_0)

## The construction phase

num_iter = 10000
num_hidden_neurons = [50]
num_hidden_layers = np.size(num_hidden_neurons)


with tf.variable_scope('dnn'):

    previous_layer = x_0

    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l],activation=tf.nn.sigmoid)
        previous_layer = current_layer

    dnn_output = tf.layers.dense(previous_layer, matrix_size)

with tf.name_scope('loss'):
    print("dnn_output = ", dnn_output)
    
    x_trial = tf.transpose(dnn_output)
    print("x_trial = ", x_trial)
    
    temp1 = (tf.tensordot(tf.transpose(x_trial), x_trial, axes=1)*A)
    temp2 = (1- tf.tensordot(tf.transpose(x_trial), tf.tensordot(A, x_trial, axes=1), axes=1))*np.eye(matrix_size)
    func = tf.tensordot((temp1-temp2), x_trial, axes=1)
    
    print(temp1)
    print(temp2)
    print(func)
    
    func = tf.transpose(func)
    x_trial = tf.transpose(x_trial)
    
    loss = tf.losses.mean_squared_error(func, x_trial)

learning_rate = 0.001

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traning_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

g_dnn = None

losses = []

with tf.Session() as sess:
    init.run()
    for i in range(num_iter):
        sess.run(traning_op)

        if i % 100 == 0:
            l = loss.eval()
            print("Step:", i, "/",num_iter, "loss: ", l)
            losses.append(l)

    x_dnn = x_trial.eval()
x_dnn = x_dnn.T


# ## Plotting loss over time

plt.plot(losses[:5])
plt.xlabel("Iteration")
plt.ylabel("Loss")

print("Eigenvector NN = \n", (x_dnn/(x_dnn**2).sum()**0.5), "\n")

eigen_val_nn = x_dnn.T @ (start_matrix @ x_dnn) / (x_dnn.T @ x_dnn)

print("Eigenvalue NN = \n", eigen_val_nn, "\n \n")
print("Eigenvector analytic = \n", eigen_vecs)
print("\n")
print("Eigenvalues analytic = \n",eigen_vals)

