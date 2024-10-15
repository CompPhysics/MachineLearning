#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html exercisesweek41.do.txt  -->
# <!-- dom:TITLE: Exercises week 41 -->
# 

# # Exercises week 42
# 
# **October 14-18, 2024**
# 
# Date: **Deadline is Friday October 18 at midnight**
# 

# # Overarching aims of the exercises this week
# 
# This week, you will implement the entire feed-forward pass of a neural network! Next week you will compute the gradient of the network by implementing back-propagation manually, and by using autograd which does back-propagation for you (much easier!). Next week, you will also use the gradient to optimize the network with a gradient method! However, there is an optional exercise this week to get started on training the network and getting good results!
# 
# We recommend that you do the exercises this week by editing and running this notebook file, as it includes some checks along the way that you have implemented the pieces of the feed-forward pass correctly, and running small parts of the code at a time will be important for understanding the methods.
# 
# If you have trouble running a notebook, you can run this notebook in google colab instead (https://colab.research.google.com/drive/1zKibVQf-iAYaAn2-GlKfgRjHtLnPlBX4#offline=true&sandboxMode=true), an updated link will be provided on the course discord (you can also send an email to k.h.fredly@fys.uio.no if you encounter any trouble), though we recommend that you set up VSCode and your python environment to run code like this locally.
# 
# First, here are some functions you are going to need, don't change this cell. If you are unable to import autograd, just swap in normal numpy until you want to do the final optional exercise.
# 

# In[1]:


import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


# # Exercise 1
# 
# In this exercise you will compute the activation of the first layer. You only need to change the code in the cells right below an exercise, the rest works out of the box. Feel free to make changes and see how stuff works though!
# 

# In[2]:


np.random.seed(2024)

x = np.random.randn(2)  # network input. This is a single input with two features
W1 = np.random.randn(4, 2)  # first layer weights


# **a)** Given the shape of the first layer weight matrix, what is the input shape of the neural network? What is the output shape of the first layer?
# 

# **b)** Define the bias of the first layer, `b1`with the correct shape. (Run the next cell right after the previous to get the random generated values to line up with the test solution below)
# 

# In[3]:


b1 = ...


# **c)** Compute the intermediary `z1` for the first layer
# 

# In[4]:


z1 = ...


# **d)** Compute the activation `a1` for the first layer using the ReLU activation function defined earlier.
# 

# In[5]:


a1 = ...


# Confirm that you got the correct activation with the test below. Make sure that you define `b1` with the randn function right after you define `W1`.
# 

# In[6]:


sol1 = np.array([0.60610368, 4.0076268, 0.0, 0.56469864])

print(np.allclose(a1, sol1))


# # Exercise 2
# 
# Now we will add a layer to the network with an output of length 8 and ReLU activation.
# 
# **a)** What is the input of the second layer? What is its shape?
# 
# **b)** Define the weight and bias of the second layer with the right shapes.
# 

# In[ ]:


W2 = ...
b2 = ...


# **c)** Compute the intermediary `z2` and activation `a2` for the second layer.
# 

# In[ ]:


z2 = ...
a2 = ...


# Confirm that you got the correct activation shape with the test below.
# 

# In[ ]:


print(
    np.allclose(np.exp(len(a2)), 2980.9579870417283)
)  # This should evaluate to True if a2 has the correct shape :)


# # Exercise 3
# 
# We often want our neural networks to have many layers of varying sizes. To avoid writing very long and error-prone code where we explicitly define and evaluate each layer we should keep all our layers in a single variable which is easy to create and use.
# 
# **a)** Complete the function below so that it returns a list `layers` of weight and bias tuples `(W, b)` for each layer, in order, with the correct shapes that we can use later as our network parameters.
# 

# In[ ]:


def create_layers(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = ...
        b = ...
        layers.append((W, b))

        i_size = layer_output_size
    return layers


# **b)** Comple the function below so that it evaluates the intermediary `z` and activation `a` for each layer, with ReLU actication, and returns the final activation `a`. This is the complete feed-forward pass, a full neural network!
# 

# In[ ]:


def feed_forward_all_relu(layers, input):
    a = input
    for W, b in layers:
        z = ...
        a = ...
    return a


# **c)** Create a network with input size 8 and layers with output sizes 10, 16, 6, 2. Evaluate it and make sure that you get the correct size vectors along the way.
# 

# In[ ]:


input_size = ...
layer_output_sizes = [...]

x = np.random.rand(input_size)
layers = ...
predict = ...
print(predict)


# **d)** Why is a neural network with no activation functions always mathematically equivelent to a neural network with only one layer?
# 

# # Exercise 4 - Custom activation for each layer
# 

# So far, every layer has used the same activation, ReLU. We often want to use other types of activation however, so we need to update our code to support multiple types of activation functions. Make sure that you have completed every previous exercise before trying this one.
# 

# **a)** Complete the `feed_forward` function which accepts a list of activation functions as an argument, and which evaluates these activation functions at each layer.
# 

# In[ ]:


def feed_forward(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = ...
        a = ...
    return a


# **b)** You are now given a list with three activation functions, two ReLU and one sigmoid. (Don't call them yet! you can make a list with function names as elements, and then call these elements of the list later. If you add other functions than the ones defined at the start of the notebook, make sure everything is defined using autograd's numpy wrapper, like above, since we want to use automatic differentiation on all of these functions later.)
# 
# Evaluate a network with three layers and these activation functions.
# 

# In[ ]:


network_input_size = ...
layer_output_sizes = [...]
activation_funcs = [ReLU, ReLU, sigmoid]
layers = ...

x = np.random.randn(network_input_size)
feed_forward(x, layers, activation_funcs)


# **c)** How does the output of the network change if you use sigmoid in the hidden layers and ReLU in the output layer?
# 

# # Exercise 5 - Processing multiple inputs at once
# 

# So far, the feed forward function has taken one input vector as an input. This vector then undergoes a linear transformation and then an element-wise non-linear operation for each layer. This approach of sending one vector in at a time is great for interpreting how the network transforms data with its linear and non-linear operations, but not the best for numerical efficiency. Now, we want to be able to send many inputs through the network at once. This will make the code a bit harder to understand, but it will make it faster, and more compact. It will be worth the trouble.
# 
# To process multiple inputs at once, while still performing the same operations, you will only need to flip a couple things around.
# 

# **a)** Complete the function `create_layers_batch` so that the weight matrix is the transpose of what it was when you only sent in one input at a time.
# 

# In[ ]:


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = ...
        b = ...
        layers.append((W, b))

        i_size = layer_output_size
    return layers


# **b)** Make a matrix of inputs with the shape (number of features, number of inputs), you choose the number of inputs and features per input. Then complete the function `feed_forward_batch` so that you can process this matrix of inputs with only one matrix multiplication and one broadcasted vector addition per layer. (Hint: You will only need to swap two variable around from your previous implementation, but remember to test that you get the same results for equivelent inputs!)
# 

# In[ ]:


inputs = np.random.rand(1000, 4)


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = ...
        a = ...
    return a


# **c)** Create and evaluate a neural network with 4 inputs and layers with output sizes 12, 10, 3 and activations ReLU, ReLU, softmax.
# 

# In[ ]:


network_input_size = ...
layer_output_sizes = [...]
activation_funcs = [...]
layers = create_layers_batch(network_input_size, layer_output_sizes)

x = np.random.randn(network_input_size)
feed_forward_batch(inputs, layers, activation_funcs)


# You should use this batched approach moving forward, as it will lead to much more compact code. However, remember that each input is still treated separately, and that you will need to keep in mind the transposed weight matrix and other details when implementing backpropagation.
# 

# # Exercise 6 - Predicting on real data
# 

# You will now evaluate your neural network on the iris data set (https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html).
# 
# This dataset contains data on 150 flowers of 3 different types which can be separated pretty well using the four features given for each flower, which includes the width and length of their leaves. You are will later train your network to actually make good predictions.
# 

# In[ ]:


iris = datasets.load_iris()

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)


# In[ ]:


inputs = iris.data

# Since each prediction is a vector with a score for each of the three types of flowers,
# we need to make each target a vector with a 1 for the correct flower and a 0 for the others.
targets = np.zeros((len(iris.data), 3))
for i, t in enumerate(iris.target):
    targets[i, t] = 1


def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)


# **a)** What should the input size for the network be with this dataset? What should the output size of the last layer be?
# 

# **b)** Create a network with two hidden layers, the first with sigmoid activation and the last with softmax, the first layer should have 8 "nodes", the second has the number of nodes you found in exercise a). Softmax returns a "probability distribution", in the sense that the numbers in the output are positive and add up to 1 and, their magnitude are in some sense relative to their magnitude before going through the softmax function. Remember to use the batched version of the create_layers and feed forward functions.
# 

# In[ ]:


...
layers = ...


# **c)** Evaluate your model on the entire iris dataset! For later purposes, we will split the data into train and test sets, and compute gradients on smaller batches of the training data. But for now, evaluate the network on the whole thing at once.
# 

# In[ ]:


predictions = feed_forward_batch(inputs, layers, activation_funcs)


# **d)** Compute the accuracy of your model using the accuracy function defined above. Recreate your model a couple times and see how the accuracy changes.
# 

# In[ ]:


print(accuracy(predictions, targets))


# # Exercise 7 - Training on real data (Optional)
# 
# To be able to actually do anything useful with your neural network, you need to train it. For this, we need a cost function and a way to take the gradient of the cost function wrt. the network parameters. The following exercises guide you through taking the gradient using autograd, and updating the network parameters using the gradient. Feel free to implement gradient methods like ADAM if you finish everything.
# 

# Since we are doing a classification task with multiple output classes, we use the cross-entropy loss function, which can evaluate performance on classification tasks. It sees if your prediction is "most certain" on the correct target.
# 

# In[ ]:


def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))


def cost(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target)


# To improve our network on whatever prediction task we have given it, we need to use a sensible cost function, take the gradient of that cost function with respect to our network parameters, the weights and biases, and then update the weights and biases using these gradients. To clarify, we need to find and use these
# 
# $$
# \frac{\partial C}{\partial W}, \frac{\partial C}{\partial b}
# $$
# 

# Now we need to compute these gradients. This is pretty hard to do for a neural network, we will use most of next week to do this, but we can also use autograd to just do it for us, which is what we always do in practice. With the code cell below, we create a function which takes all of these gradients for us.
# 

# In[ ]:


from autograd import grad


gradient_func = grad(
    cost, 1
)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers


# **a)** What shape should the gradient of the cost function wrt. weights and biases be?
# 
# **b)** Use the `gradient_func` function to take the gradient of the cross entropy wrt. the weights and biases of the network. Check the shapes of what's inside. What does the `grad` func from autograd actually do?
# 

# In[ ]:


layers_grad = gradient_func(
    inputs, layers, activation_funcs, targets
)  # Don't change this


# **c)** Finish the `train_network` function.
# 

# In[ ]:


def train_network(
    inputs, layers, activation_funcs, targets, learning_rate=0.001, epochs=100
):
    for i in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        for (W, b), (W_g, b_g) in zip(layers, layers_grad):
            W -= ...
            b -= ...


# **e)** What do we call the gradient method used above?
# 

# **d)** Train your network and see how the accuracy changes! Make a plot if you want.
# 

# In[ ]:


...


# **e)** How high of an accuracy is it possible to acheive with a neural network on this dataset, if we use the whole thing as training data?
# 
