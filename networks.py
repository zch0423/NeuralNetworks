'''
filename: networks.py
content:
    definition of neural network by preceptron model and backpropagation algorithm
'''
import numpy as np
from _functions import ACTIVATIONS, DERIVATIVES, LOSS
from _solver import AdamOptimizer
from tools import gen_batches, sparse_dot

class BasePerceptron:
    '''
    base class
    hidden_layer_sizes: (100, )
    activation: activation function--relu sigmoid softmax tanh
    out_activation: activation function for output--here is softmax
    learning_rate: learning rate for update
    max_iter: maximum number of iteration
    tol: tolerance for the model when loss score is not improving
    loss: loss function--squared loss
    solver: here we use ADAM algorithm
    batch_size: 'auto' or int--
        minibatches for solver
        auto: batch_size = min(200, n_samples)
    alpha: L2 penalty parameter

    beta1    parameters for ADAM algorithm
    beta2    parameters for ADAM algorithm
    epsilon  parameters for ADAM algorithm

    n_iter_no_change: max number of iterations to not meet acquired tol improvement
    '''

    def __init__(self, hidden_layer_sizes, activation, 
                out_activation, learning_rate, max_iter, tol,
                loss, solver, batch_size, alpha, beta1, beta2, epsilon,
                n_iter_no_change):
        self.hidden_layer_sizes = hidden_layer_sizes  # like(100, )
        self.activation = activation
        self.out_activation = out_activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.loss = loss
        self.solver = solver
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

    def _initialize(self, y, layer_units):
        # initialize parameters--allocate weights
        self.n_iter_ = 0  # 迭代数
        self.t_ = 0  # for learning rate update
        # self.n_outputs_ = y.shape[1]
        self.n_layers_ = len(layer_units)  # 层数
        # initialize coefficients and intercept
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(self.n_layers_-1):
            coef_init, intercept_init = self._init_coef(layer_units[i], layer_units[i+1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        # for solver
        self.loss_curve_ = []
        self._no_improvement_count = 0
        self.best_loss_ = np.inf

    def _init_coef(self, n_in, n_out):
        # Glorot initialization
        factor = 6
        if self.activation == "sigmoid":
            factor = 2
        init_bound = np.sqrt(factor/(n_in+n_out))
        # generate weights and bias
        # uniform random numbers of size (n_in, n_out)
        coef_init = np.random.uniform(-init_bound, init_bound,
                                    (n_in, n_out))
        intercept_init = np.random.uniform(-init_bound, init_bound,
                                    n_out)
        return coef_init, intercept_init

    def _update_no_improvement_count(self, X_val, y_val):
        # check for improvement
        if self.loss_curve_[-1]>self.best_loss_ - self.tol:
            # best_loss initialized with infinity
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0  # reset
        if self.loss_curve_[-1] < self.best_loss_:
            self.best_loss_ = self.loss_curve_[-1]

    def _forward_pass(self,activations):
        '''
        forward pass
        computing the values in the hidden layers and the output

        activations:
            ith elemen--values of ith layer
        '''
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_-1):
            # iterate hidden layer
            activations[i+1] = sparse_dot(activations[i], self.coefs_[i])
            activations[i+1] += self.intercepts_[i]
            if(i+1)!=(self.n_layers_-1):
                # hidden layer
                activations[i+1] = hidden_activation(activations[i+1])
        # last layer
        output_activation = ACTIVATIONS[self.out_activation]
        activations[i+1] = output_activation(activations[i+1])
        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        '''
        compute gradient of loss
        backpropagation for one layer
        '''
        coef_grads[layer] = sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += (self.alpha*self.coefs_[layer])
        coef_grads[layer] /= n_samples
        intercept_grads[layer] = np.mean(deltas[layer], 0)
        return coef_grads, intercept_grads

    def _backprop(self, X, y, activations, deltas,
                   coef_grads, intercept_grads):
        '''
        calculate loss function and derivatives
        WRT weights and bias

        activations:
            the ith element represents values of the ith layer
        deltas:
            ith element represents difference between i+1 layer and backpropagated error
            gradients of loss WRT z in each layer,
            where z=wx+b is the value of a layer before activation function
        coef_grads:
            changes to update the coef parameters of ith layer
        intercept_grads:
            changes to update the intercept parameters
        
        returns
        loss
        coef_grads
        intercept_grads
        '''
        n_samples = X.shape[0]
        # forward pass
        activations = self._forward_pass(activations)

        loss = LOSS[self.loss](y, activations[-1])
        # add L2 regularization
        # ravel 多维数组降为一维
        values = np.sum(
            np.array([np.dot(coef.ravel(), coef.ravel()) for coef in self.coefs_]))
        loss += (0.5*self.alpha)*values/n_samples

        # back propagate
        last = self.n_layers_ -2
        deltas[last] = activations[-1]-y
        # compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)
        
        # hidden layers
        for i in range(self.n_layers_-2,0,-1):
            deltas[i-1] = sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i-1])
            coef_grads, intercept_grads = self._compute_loss_grad(
                i-1, n_samples, activations, deltas, coef_grads, intercept_grads)

        return loss, coef_grads, intercept_grads

    def _fit(self, X, y, activations, deltas, coef_grads,
             intercept_grads, layer_units):
        # 拟合
        params = self.coefs_ + self.intercepts_
        if self.solver == "adam":
            # only to be adam
            self._solver = AdamOptimizer(
                params, self.learning_rate, self.beta1, 
                self.beta2, self.epsilon)
        else:
            raise TypeError("solver must be 'adam' currently")

        X_val = None
        y_val = None
        n_samples = X.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                accumulated_loss = 0.0
                # 生成切片
                for batch_slice in gen_batches(n_samples, batch_size):
                    # activations [X]+[None]*(len(layer_units)-1)
                    activations[0] = X[batch_slice]

                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss*(batch_slice.stop-
                                                    batch_slice.start)
                    #update weights
                    grads = coef_grads + intercept_grads
                    #learning rate also updated
                    self._solver.update_params(grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss/X.shape[0]
                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                # test output
                print("Iteration %d,loss=%.8f"%(self.n_iter_, self.loss_))

                # update no improvement count
                self._update_no_improvement_count(X_val, y_val)

                if self._no_improvement_count>self.n_iter_no_change:
                    # stop or decrease learning rate
                    msg = ("Loss did not improve more than %f for %d consecutive epochs"%(
                        self.tol, self.n_iter_no_change))
                    print(msg," Stopping")

                if self.n_iter_ == self.max_iter:
                    print("Maximum iterations %d reached and not converged"%self.max_iter)
        except KeyboardInterrupt:
            print("Interrupted by user")

    def fit(self, X, y):
        '''
        fit the model using training set X and y
        X y are lists
        return a trained model
        '''
        # already assured right input
        # leave out input validation
        hidden_layer_sizes = list(self.hidden_layer_sizes)
        # 样本数  特征数
        n_samples, n_features = X.shape
        # 把y转变为二维向量
        if y.ndim == 1:
            y = y.reshape((-1,1))
        # 输出节点个数
        self.n_outputs = y.shape[1]
        # layer_units [input, hidden1, hidden2, ..., output]
        layer_units = ([n_features]+hidden_layer_sizes+[self.n_outputs])
        # 初始化权重
        self._initialize(y, layer_units)

        #随机梯度下降法 ADAM(adaptive moment estimation)
        # initialize
        activations = [X]+[None]*(len(layer_units)-1)
        deltas = [None]* (len(activations)-1)

        coef_grads = []
        intercept_grads = []
        for n_in, n_out in zip(layer_units[:-1], layer_units[1:]):
            coef_grads.append(np.empty((n_in, n_out)))
            intercept_grads.append(np.empty(n_out))
        
        # fit
        self._fit(X, y, activations, deltas, coef_grads,
                  intercept_grads, layer_units)

        return self
        
    def _predict(self, X):
        # predict
        hidden_layer_sizes = list(self.hidden_layer_sizes)
        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs]
        # initialize layers
        activations = [X]
        for i in range(self.n_layers_-1):
            activations.append(np.empty((X.shape[0],layer_units[i+1])))
        # forward pass
        self._forward_pass(activations)
        y_pred = activations[-1]
        return y_pred


class NeuralNetworkClassifier(BasePerceptron):
    '''
    implementation of neural network perceptron model
    inherited from BasePerceptron 

    parameters
    hidden_layer_sizes: (100, )
    activation: activation function--relu sigmoid softmax tanh
    out_activation: activation function for output--here is softmax
    learning_rate: learning rate for update
    max_iter: maximum number of iteration
    tol: tolerance for the model when loss score is not improving
    loss: loss function--squared loss
    solver: here we use ADAM algorithm
    batch_size: 'auto' or int--
        minibatches for solver
        auto: batch_size = min(200, n_samples)
    alpha: L2 penalty parameter

    beta1    parameters for ADAM algorithm
    beta2    parameters for ADAM algorithm
    epsilon  parameters for ADAM algorithm

    n_iter_no_change: max number of iterations to not meet acquired tol improvement
    '''
    def __init__(self, hidden_layer_sizes=(100,), activation="sigmoid",
                 out_activation="softmax", learning_rate=0.001, max_iter=200,
                 tol=1e-4, loss="squared_loss", solver="adam", 
                 batch_size="auto", alpha=0.0001, beta1=0.9, 
                 beta2=0.999, epsilon=1e-8, n_iter_no_change=10):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, 
                         activation=activation,
                         out_activation=out_activation, 
                         learning_rate=learning_rate,
                         max_iter=max_iter, tol=tol,
                         loss=loss, solver=solver,
                         batch_size=batch_size, alpha=alpha,
                         beta1=beta1, beta2=beta2, epsilon=epsilon,
                         n_iter_no_change=n_iter_no_change)

    def predict(self, X):
        '''
        predict X
        return:
            an array of labels
            for multi outputs--return the index with highest value
        '''
        y_pred = self._predict(X)
        if self.n_outputs == 1:
            y_pred = y_pred.ravel()
        else:
            # 选取概率最高的返回index值
            y_pred = predict_transform(y_pred)
        return y_pred


def predict_transform(y):
    '''
    transformation of multi outputs
    pick the node with highest value
    return index
    '''
    return np.array([np.argmax(row) for row in y])

