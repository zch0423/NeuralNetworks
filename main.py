'''
filename: main.py
content: main function
'''
import numpy as np
import visualization as vis
from tools import array2Label, train_test_split, accuracy_score
from networks import NeuralNetworkClassifier
from preprocess import dataprocess


filepath = "datapca_10.csv"
X, y = dataprocess(filepath)

def oneFit(X, y, activation="relu", hidden_layer=(30, 30), test_size=0.2):
    '''
    process of one fit
    return accuracy
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_label = array2Label(y_test)  # transform 2d array into 1d labels
    # 激活函数使用relu
    nn = NeuralNetworkClassifier(hidden_layer_sizes=hidden_layer, activation=activation)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    # random guess 0.006
    return accuracy_score(y_label, y_pred)


def multiFit(X, y, n=20 ,activation="relu", hidden_layer=(30, 30), test_size=0.2):
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

def layerEffect(max_layers=5, max_nodes=40):
    '''
    test the effect of different hidden layer sizes
    including the number of layers and the number of nodes in each layer
    using relu as activation function 
    '''
    accuracy_mean = np.zeros((max_layers, max_nodes))
    accuracy_std = np.zeros((max_layers, max_nodes))
    for n_layer in range(max_layers):
        temp_mean = np.zeros(max_nodes)
        temp_std = np.zeros(max_nodes)
        for n_node in range(max_nodes): 
            layers = [n_node+1 for i in range(n_layer+1)]
            mean, std = multiFit(X, y, hidden_layer=layers, activation="relu")
            temp_mean[n_node] = mean
            temp_std[n_node] = std
            # output
            print("layer num:", n_layer+1,"node num:", n_node+1, "--training finished")
        accuracy_mean[n_layer] = temp_mean
        accuracy_std[n_layer] = temp_std
    return accuracy_mean, accuracy_std


def testSizeEffect(low=0.05, high=0.3, n=6):
    '''
    test effect of different test sizes when splitting the data set
    n test size from low to high  [low, high]
    '''
    testSizes = np.linspace(low, high, n)
    accuracy_mean = np.zeros(n)
    accuracy_std = np.zeros(n)
    for size in testSizes:
        result = multiFit(X, y, test_size=size)



def main():
    result_layer = layerEffect()
    vis.layerEffect3D(result_layer[0], save=True)  # True to save
    #TODO
    '''
    some analysis about the effect of 
        number of layers 
        number of nodes for each layer
        different activation
        different pca size
    '''
main()
