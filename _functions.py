'''
filename: functions.py
content: 
    definition of activition functions
    loss function
    derivative of activation functions

'''

import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def tanh(X):
    return np.tanh(X)

def relu(X):
    # 0-infinity y=x, else y=0
    return np.clip(X, 0, np.finfo(X.dtype).max)

def softmax(X):
    X = np.exp(X)
    temp = np.sum(X, axis=1)
    for i in range(len(temp)):
        X[i] /= temp[i]
    return X

def inplace_dsigmoid(Z ,delta):
    '''
    INPUT
    Z: output of former activation function
    delta: array waited to be updated

    derivative: Z*(1-Z)
    inplace change for delta
    '''
    delta *= Z
    delta *= (1-Z)

def inplace_dtanh(Z, delta):
    '''
    INPUT
    Z: output of former activation function
    delta: array waited to be updated
    
    derivative:1-Z^2
    inplace change for delta
    '''
    delta *= (1 - Z ** 2)

def inplace_drelu(Z, delta):
    '''
    INPUT
    Z: output of former activation function
    delta: array waited to be updated

    derivative: 0 for x==0 and 1 for x>0
    inplace change for delta
    '''
    # Z==0 returns an array of bool type
    delta[Z==0] = 0

def squared_loss(y_true, y_pred):
    '''
    loss function
    sum((y_true-y_pred)^2)/2n
    '''
    return np.mean((y_true-y_pred)**2)/2

ACTIVATIONS = {"relu": relu, 
               "tanh":tanh, 
               "sigmoid": sigmoid, 
               "softmax": softmax}
DERIVATIVES = {"relu": inplace_drelu, 
               "tanh": inplace_dtanh,
               "sigmoid": inplace_dsigmoid}

LOSS = {"squared_loss": squared_loss}
