'''
filename: networks.py
content:
    definition of neural network by preceptron model and backpropagation algorithm
'''
import numpy as np
from tools import ACTIVATIONS, DERIVATIVES, LOSS

class BasePerceptron:
    '''
    abstract base class
    '''

    def __init__(self, hidden_layer_sizes, activation, 
                learning_rate, max_iter, loss):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss = loss

    def _initialize(self, y, layer_units):
        # some initialization before self.fit
        # y is a 2D array [y1;...;yn]
        #TODO
        pass


