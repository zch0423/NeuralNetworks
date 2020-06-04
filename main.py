'''
filename: main.py
content: main function
'''
import numpy as np
from tools import array2Label, train_test_split, accuracy_score
from networks import NeuralNetworkClassifier
from preprocess import dataprocess


filepath = "datapca_10.csv"
X, y = dataprocess(filepath)

def oneFit(X, y, activation="sigmoid", hidden_layer=(100,), test_size=0.3):
    '''
    process of one fit
    return accuracy
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_label = array2Label(y_test)  # transform 2d array into 1d labels
    # 激活函数使用relu
    nn = NeuralNetworkClassifier(hidden_layer_sizes=hidden_layer,
                                 max_iter=1000, activation=activation)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    # random guess 0.006
    return accuracy_score(y_label, y_pred)


def multiFit(X, y, n=100 ,activation="sigmoid", hidden_layer=(300, 300), test_size=0.3):
    '''
    fit the model for n times and return mean accuracy and standard error
    INPUT
    X, y
    n: number of iterations
    activation: activation function
    hidden_layer: layer information for the model
    test_size
    RETURNS
    mean std
    '''
    accuracy = np.array([oneFit(X, y, activation=activation,
                                hidden_layer=hidden_layer, test_size=test_size)
                        for i in range(n)])
    mean = np.mean(accuracy)  # 1D
    std = np.std(accuracy, ddof=1)
    return mean, std


def activationEffect():
    '''
    test difference between activation functions
    '''
    pass

def layerEffect():
    '''
    test the effect of different hidden layer sizes
    including the number of layers and the number of nodes in each layer
    '''
    pass

def testSizeEffect():
    '''
    test effect of different test sizes when splitting the data set
    '''
    pass

def PCAEffect():
    '''
    test effect of different pca dimensions chosen
    '''
    pass

def main():
    print(oneFit(X, y, activation="relu"))
    
    #TODO
    '''
    some analysis about the effect of 
        number of layers 
        number of nodes for each layer
        different activation
        different pca size
    '''
main()
