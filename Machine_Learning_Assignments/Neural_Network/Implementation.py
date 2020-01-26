import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import scipy.io
import pylab
from Model import *

def get_layers_dims(input_size, d, h, output_size):
    layers_dims = list()
    layers_dims.append(input_size)
    counter = 0
    while counter < d:
        layers_dims.append(h)
        counter += 1
    layers_dims.append(output_size)
    return layers_dims

def get_data(file, data):
    temp = scipy.io.loadmat(file)
    x, y = [], []
    for i in range(2):
        for val in temp[data + str(i)]:
            x.append(val)
            y.append(i)

    return np.array(x), np.array(y)

np.random.seed(1)

train_x, train_y = get_data('mnist_all.mat', 'train')
test_x, test_y = get_data('mnist_all.mat', 'test')

train_x = train_x / 255.
test_x = test_x / 255.
### CONSTANTS DEFINING THE MODEL ####
n_x = 784     # num_px * num_px * 3
n_h = 7
n_y = 1

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0003, num_iterations = 1500, print_cost = True):
    np.random.seed(1)
    costs = []                         # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)

    AL, caches = L_model_forward(X, parameters)
    # Backward propagation.
    grads = L_model_backward(AL, Y, caches)
    # Update parameters.
    parameters = update_parameters(parameters, grads, learning_rate)

    for i in range(1, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

    # ##USING STOCHASTIC GRADIENT DESCENT
    ##batch_size = 1000
    # # Loop (gradient descent)
    # for i in range(0, num_iterations):
    #     counter = 0
    #     while counter < len(X):
    #         AL, caches = L_model_forward(X.T[counter:counter + batch_size].T, parameters)
    #         # cost = compute_cost(AL, Y)
    #
    #         # Backward propagation.
    #         grads = L_model_backward(AL, Y[counter:counter + batch_size], caches)
    #
    #         for grad in grads:
    #             grads[grad] /= batch_size
    #
    #         # Update parameters.
    #         parameters = update_parameters(parameters, grads, learning_rate)
    #         print(grads["dA1"][0][1])
    #         counter += batch_size
    
    return parameters, grads


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p

# #Q2 - Part C
input_size = 784
d = 2
h = 7
output_size = 1
layers_dims = get_layers_dims(input_size, d, h, output_size)
parameters, grads = L_layer_model(train_x.T, train_y, layers_dims, num_iterations = 1500, print_cost = True)
pred_train = predict(train_x.T, train_y, parameters)
pred_test = predict(test_x.T, test_y, parameters)

#Q2 - Part D
d = 1
input_size = 784
output_size = 1
h = 7
norms = list()
while d <= 10:
    layers_dims = get_layers_dims(input_size, d, h, output_size)
    parameters, grads = L_layer_model(train_x.T, train_y, layers_dims, num_iterations=1, print_cost = True)
    norm = np.linalg.norm(grads['dW1'], ord='fro')
    print(norm)
    norms.append(norm)
    d += 1

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(2, 2, 1)
line, = ax.plot(norms, color='black', lw=1)
ax.set_yscale('log')
plt.xlabel('Hidden Layers')
plt.ylabel('Log Frobenius Norm of dW1')
pylab.show()