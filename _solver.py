'''
filename: _solver.py
content: definition of neural network solver 
'''
import numpy as np

class AdamOptimizer:
    '''
    stochastic gradient descent with adaptive moment estimation(adam)
    credit to Kingma, Diederik and Jimmy Ba

    parameters
    params:
        coefs_+intercepts_
    learning_rate:
        initialize learning rate
    beta1:
        指数衰减率 first moment vector
    beta2:
        second moment vector
    epsilon
    '''
    def __init__(self, params, learning_rate_init=0.001, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)
        self.t = 0
        # initialize
        self.params = [param for param in params]
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        '''
        update parameters with gradients
        grads: gradients of coefs and interceptions
        '''
        self.t += 1
        self.ms = [self.beta1*m + (1-self.beta1)*grad
                    for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta2*v + (1-self.beta2)*(grad**2)
                    for v, grad in zip(self.vs, grads)]
        temp = np.sqrt(1-self.beta2**self.t)/ (1-self.beta1**self.t)
        self.learning_rate = self.learning_rate_init*temp
        updates = [-self.learning_rate*m / (np.sqrt(v)+self.epsilon)
                    for m, v in zip(self.ms, self.vs)]
        
        for param, update in zip(self.params, updates):
            param += update