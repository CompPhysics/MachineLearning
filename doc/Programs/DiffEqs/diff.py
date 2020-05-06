import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.reset_default_graph()
import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras import optimizers
from keras import backend as K

## Creates a trial function g = y0 + x * N(x, P)
def trial_func(x, y, y0 = 1):
    func = tf.exp(-x)*y0 + x * y  
    return func

## Computes Right Side of differential eq; -k/m * g
def right_side(trial, k = 1, m = 1):
    return -trial

## Here we define the loss function
def loss_wrapper(input_tensor):
    def loss_function(y, y_pred):
        ## Find the trial solution and right-side
        trial = trial_func(input_tensor, y_pred)
        right = right_side(trial)   
        #  For coupled second-order we may need to have two loss function
        left = tf.gradients(trial, input_tensor) 
        loss = tf.reduce_mean(tf.math.squared_difference(left, right))    
        return loss
        
    return loss_function

def create_input_data(a = 0, b = 5, n = 100):
    input_data = np.linspace(a,b,n)
    input_data = input_data.reshape(1,n)
    return input_data

def create_model(data, n_inputs, n_hidden_layer = 50):
    input_tensor = Input(shape=(n_inputs,))
    hidden = Dense(30, activation='tanh',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(input_tensor)
    hidden2 = Dense(200, activation='tanh',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(hidden)
    hidden3 = Dense(50, activation='tanh',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(hidden2)
    hidden4 = Dense(30, activation='tanh',
                   kernel_initializer='random_uniform', bias_initializer='zeros')(hidden3)
    out = Dense(n_inputs)(hidden2)
    model = Model(input_tensor, out)
    
    sgd = optimizers.SGD(lr=0.001, decay = .1)
    model.compile(loss=loss_wrapper(input_tensor), optimizer='sgd')
    model.fit(data, np.zeros((data.shape[0])), epochs = 5000)
    
    res = model.predict(data)
    
    del model
    
    return res

data = create_input_data(0, 1, 10)
shape = data.shape
res = create_model(data, shape[1], 50)


results = trial_func(data[0], res)
euler = euler_cromer(tf = 1, gam = 0, mass = 1)
plt.plot(data[0], results[0], label = 'Neural Differential Equation')
plt.plot(euler[0], np.exp(-euler[0]), label = 'Euler-Cromer Method (Numerical)')
plt.legend()
plt.title('Neural ODE vs Analytical Method')
plt.xlabel('Time')
plt.xlabel('X-Position')
