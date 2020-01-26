import numpy as np

np.random.seed(1)

def sigmoid(X):
    return 1 / (1 + np.exp(-X)), X

def dsigmoid(x):
    sigmoid_x, x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_activation_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L + 1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)

    return A, caches

def compute_cost(AL, Y):
    # Compute loss from aL and y.
    cost = -(np.dot(Y, np.transpose(np.log(AL))) + np.dot(1 - Y, np.transpose(np.log(1 - AL))))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache

    dZ = dA * dsigmoid(activation_cache)

    A_prev, W, b = linear_cache

    dW = np.dot(dZ, np.transpose(A_prev))
    db = np.dot(dZ, np.ones((dZ.shape[1], 1)))
    dA_prev = np.dot(np.transpose(W), dZ)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * grads["dW" + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate * grads["db" + str(l + 1)])
    return parameters