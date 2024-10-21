#!/usr/bin/env python
# coding: utf-8

# # Exercises week 43
# 
# **October 18-25, 2024**
# 
# Date: **Deadline is Friday October 25 at midnight**
# 

# # Overarching aims of the exercises this week
# 
# The aim of the exercises this week is to train the neural network you implemented last week.
# 
# To train neural networks, we use gradient descent, since there is no analytical expression for the optimal parameters. This means you will need to compute the gradient of the cost function wrt. the network parameters. And then you will need to implement some gradient method.
# 
# You will begin by computing gradients for a network with one layer, then two layers, then any number of layers. Keeping track of the shapes and doing things step by step will be very important this week.
# 
# We recommend that you do the exercises this week by editing and running this notebook file, as it includes some checks along the way that you have implemented the neural network correctly, and running small parts of the code at a time will be important for understanding the methods. If you have trouble running a notebook, you can run this notebook in google colab instead(https://colab.research.google.com/drive/1FfvbN0XlhV-lATRPyGRTtTBnJr3zNuHL#offline=true&sandboxMode=true), though we recommend that you set up VSCode and your python environment to run code like this locally.
# 
# First, some setup code that you will need.
# 

# In[1]:


import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mse(predict, target):
    return np.mean((predict - target) ** 2)


# # Exercise 1 - Understand the feed forward pass
# 
# **a)** Complete last weeks' mandatory exercises if you haven't already.
# 

# # Exercise 2 - Gradient with one layer using autograd
# 
# For the first few exercises, we will not use batched inputs. Only a single input vector is passed through the layer at a time.
# 
# In this exercise you will compute the gradient of a single layer. You only need to change the code in the cells right below an exercise, the rest works out of the box. Feel free to make changes and see how stuff works though!
# 

# **a)** If the weights and bias of a layer has shapes (10, 4) and (10), what will the shapes of the gradients of the cost function wrt. these weights and this bias be?
# 

# **b)** Complete the feed_forward_one_layer function. It should use the sigmoid activation function. Also define the weigth and bias with the correct shapes.
# 

# In[2]:


def feed_forward_one_layer(W, b, x):
    z = ...
    a = ...
    return a


def cost_one_layer(W, b, x, target):
    predict = feed_forward_one_layer(W, b, x)
    return mse(predict, target)


x = np.random.rand(2)
target = np.random.rand(3)

W = ...
b = ...


# **c)** Compute the gradient of the cost function wrt. the weigth and bias by running the cell below. You will not need to change anything, just make sure it runs by defining things correctly in the cell above. This code uses the autograd package which uses backprogagation to compute the gradient!
# 

# In[3]:


autograd_one_layer = grad(cost_one_layer, [0, 1])
W_g, b_g = autograd_one_layer(W, b, x, target)
print(W_g, b_g)


# # Exercise 3 - Gradient with one layer writing backpropagation by hand
# 
# Before you use the gradient you found using autograd, you will have to find the gradient "manually", to better understand how the backpropagation computation works. To do backpropagation "manually", you will need to write out expressions for many derivatives along the computation.
# 

# We want to find the gradient of the cost function wrt. the weight and bias. This is quite hard to do directly, so we instead use the chain rule to combine multiple derivatives which are easier to compute.
# 
# $$
# \frac{dC}{dW} = \frac{dC}{da}\frac{da}{dz}\frac{dz}{dW}
# $$
# 
# $$
# \frac{dC}{db} = \frac{dC}{da}\frac{da}{dz}\frac{dz}{db}
# $$
# 

# **a)** Which intermediary results can be reused between the two expressions?
# 

# **b)** What is the derivative of the cost wrt. the final activation? You can use the autograd calculation to make sure you get the correct result. Remember that we compute the mean in mse.
# 

# In[ ]:


z = W @ x + b
a = sigmoid(z)

predict = a


def mse_der(predict, target):
    return ...


print(mse_der(predict, target))

cost_autograd = grad(mse, 0)
print(cost_autograd(predict, target))


# **c)** What is the expression for the derivative of the sigmoid activation function? You can use the autograd calculation to make sure you get the correct result.
# 

# In[ ]:


def sigmoid_der(z):
    return ...


print(sigmoid_der(z))

sigmoid_autograd = elementwise_grad(sigmoid, 0)
print(sigmoid_autograd(z))


# **d)** Using the two derivatives you just computed, compute this intermetidary gradient you will use later:
# 
# $$
# \frac{dC}{dz} = \frac{dC}{da}\frac{da}{dz}
# $$
# 

# In[54]:


dC_da = ...
dC_dz = ...


# **e)** What is the derivative of the intermediary z wrt. the weight and bias? What should the shapes be? The one for the weights is a little tricky, it can be easier to play around in the next exercise first. You can also try computing it with autograd to get a hint.
# 

# **f)** Now combine the expressions you have worked with so far to compute the gradients! Note that you always need to do a feed forward pass while saving the zs and as before you do backpropagation, as they are used in the derivative expressions
# 

# In[ ]:


dC_da = ...
dC_dz = ...
dC_dW = ...
dC_db = ...

print(dC_dW, dC_db)


# You should get the same results as with autograd.
# 

# In[ ]:


W_g, b_g = autograd_one_layer(W, b, x, target)
print(W_g, b_g)


# # Exercise 4 - Gradient with two layers writing backpropagation by hand
# 

# Now that you have implemented backpropagation for one layer, you have found most of the expressions you will need for more layers. Let's move up to two layers.
# 

# In[59]:


x = np.random.rand(2)
target = np.random.rand(4)

W1 = np.random.rand(3, 2)
b1 = np.random.rand(3)

W2 = np.random.rand(4, 3)
b2 = np.random.rand(4)

layers = [(W1, b1), (W2, b2)]


# In[60]:


z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)


# We begin by computing the gradients of the last layer, as the gradients must be propagated backwards from the end.
# 
# **a)** Compute the gradients of the last layer, just like you did the single layer in the previous exercise.
# 

# In[61]:


dC_da2 = ...
dC_dz2 = ...
dC_dW2 = ...
dC_db2 = ...


# To find the derivative of the cost wrt. the activation of the first layer, we need a new expression, the one furthest to the right in the following.
# 
# $$
# \frac{dC}{da_1} = \frac{dC}{dz_2}\frac{dz_2}{da_1}
# $$
# 
# **b)** What is the derivative of the second layer intermetiate wrt. the first layer activation? (First recall how you compute $z_2$)
# 
# $$
# \frac{dz_2}{da_1}
# $$
# 

# In[ ]:





# **c)** Use this expression, together with expressions which are equivelent to ones for the last layer to compute all the derivatives of the first layer.
# 
# $$
# \frac{dC}{dW_1} = \frac{dC}{da_1}\frac{da_1}{dz_1}\frac{dz_1}{dW_1}
# $$
# 
# $$
# \frac{dC}{db_1} = \frac{dC}{da_1}\frac{da_1}{dz_1}\frac{dz_1}{db_1}
# $$
# 

# In[63]:


dC_da1 = ...
dC_dz1 = ...
dC_dW1 = ...
dC_db1 = ...


# In[ ]:


print(dC_dW1, dC_db1)
print(dC_dW2, dC_db2)


# **d)** Make sure you got the same gradient as the following code which uses autograd to do backpropagation.
# 

# In[67]:


def feed_forward_two_layers(layers, x):
    W1, b1 = layers[0]
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)

    W2, b2 = layers[1]
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)

    return a2


# In[ ]:


def cost_two_layers(layers, x, target):
    predict = feed_forward_two_layers(layers, x)
    return mse(predict, target)


grad_two_layers = grad(cost_two_layers, 0)
grad_two_layers(layers, x, target)


# **e)** How would you use the gradient from this layer to compute the gradient of an even earlier layer? Would the expressions be any different?
# 

# # Exercise 5 - Gradient with any number of layers writing backpropagation by hand
# 

# Well done on getting this far! Now it's time to compute the gradient with any number of layers.
# 
# First, some code from the general neural network code from last week. Note that we are still sending in one input vector at a time. We will change it to use batched inputs later.
# 

# In[4]:


def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = W @ a + b
        a = activation_func(z)
    return a


def cost(layers, input, activation_funcs, target):
    predict = feed_forward(input, layers, activation_funcs)
    return mse(predict, target)


# You might have already have noticed a very important detail in backpropagation: You need the values from the forward pass to compute all the gradients! The feed forward method above is great for efficiency and for using autograd, as it only cares about computing the final output, but now we need to also save the results along the way.
# 
# Here is a function which does that for you.
# 

# In[5]:


def feed_forward_saver(input, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = W @ a + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a


# **a)** Now, complete the backpropagation function so that it returns the gradient of the cost function wrt. all the weigths and biases. Use the autograd calculation below to make sure you get the correct answer.
# 

# In[ ]:


def backpropagation(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der
):
    layer_inputs, zs, predict = feed_forward_saver(input, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = ...
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1]
            dC_da = ...

        dC_dz = ...
        dC_dW = ...
        dC_db = ...

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


# In[ ]:


network_input_size = 2
layer_output_sizes = [3, 4]
activation_funcs = [sigmoid, ReLU]
activation_ders = [sigmoid_der, ReLU_der]

layers = create_layers(network_input_size, layer_output_sizes)

x = np.random.rand(network_input_size)
target = np.random.rand(4)


# In[ ]:


layer_grads = backpropagation(x, layers, activation_funcs, target, activation_ders)
print(layer_grads)


# In[ ]:


cost_grad = grad(cost, 0)
cost_grad(layers, x, [sigmoid, ReLU], target)


# # Exercise 6 - Batched inputs
# 
# Make new versions of all the functions in exercise 5 which now take batched inputs instead. See last weeks exercise 5 for details on how to batch inputs to neural networks. You will also need to update the backpropogation function.
# 

# # Exercise 7 - Training
# 

# **a)** Complete exercise 6 and 7 from last week, but use your own backpropogation implementation to compute the gradient.
# 
# **b)** Use stochastic gradient descent with momentum when you train your network.
# 

# # Exercise 8 (Optional) - Object orientation
# 
# Passing in the layers, activations functions, activation derivatives and cost derivatives into the functions each time leads to code which is easy to understand in isoloation, but messier when used in a larger context with data splitting, data scaling, gradient methods and so forth. Creating an object which stores these values can lead to code which is much easier to use.
# 
# **a)** Write a neural network class. You are free to implement it how you see fit, though we strongly recommend to not save any input or output values as class attributes, nor let the neural network class handle gradient methods internally. Gradient methods should be handled outside, by performing general operations on the layer_grads list using functions or classes separate to the neural network.
# 
# We provide here a skeleton structure which should get you started.
# 

# In[ ]:


class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        pass

    def predict(self, inputs):
        # Simple feed forward pass
        pass

    def cost(self, inputs, targets):
        pass

    def _feed_forward_saver(self, inputs):
        pass

    def compute_gradient(self, inputs, targets):
        pass

    def update_weights(self, layer_grads):
        pass

    # These last two methods are not needed in the project, but they can be nice to have! The first one has a layers parameter so that you can use autograd on it
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass

