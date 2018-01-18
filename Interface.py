import numpy as np


def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    params = {}
    L = len(layers_dims)

    for l in range(1, L):
        params[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        params[f'b{l}'] = np.zeros((layers_dims[l], 1))

    return params


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z, (A, W, b)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z)), Z


def relu(Z):
    return np.maximum(0, Z), Z


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    return A, (linear_cache, activation_cache)



def L_model_forward(X, params):
    L = len(params) // 2
    A = X
    caches = []
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, params[f'W{l}'], params[f'b{l}'], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, params[f'W{L}'], params[f'b{L}'], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def predict(X, params):
    Y_, _ = L_model_forward(X, params)
    return np.vectorize(lambda x: 1 if x > 0.5 else 0)(Y_)


def accuracy(X, Y, params):
    predictions = predict(X, params)
    return (np.dot(Y, predictions.T) + np.dot(1 - Y, (1 - predictions).T)) * 100 / Y.size


def compute_cost(AL, Y):
    m = Y.shape[1]  # number of samples
    cost = -(np.dot(np.log(AL), Y.T)) + np.dot(np.log(1 - AL), (1 - Y).T)
    cost /= m
    cost = np.squeeze(cost)  # To make sure the cost's shape is what we expected
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, Z = cache

    if activation == "relu":
        dZ = dA * np.vectorize(lambda x: 1 if x > 0 else 0)(Z)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        sigm, _ = sigmoid(Z)
        dZ = dA * (sigm * (1 - sigm))
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    m = AL.shape[1]
    # Y = Y.reshape(AL.shape)  # Y is the same shape as AL
    L = len(caches)

    dAL = -(np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
    current_cache = caches[L - 1]
    grads[f'dA{L}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l];
        dA_prev, dW, db = linear_activation_backward(grads[f'dA{l+2}'], current_cache, "relu")
        grads[f'dA{l + 1}'] = dA_prev
        grads[f'dW{l + 1}'] = dW
        grads[f'db{l + 1}'] = db

    return grads


def update_parameters(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(L):
        params[f'W{l + 1}'] -= learning_rate * grads[f'dW{l + 1}']
        params[f'b{l + 1}'] -= learning_rate * grads[f'db{l + 1}']

    return params
