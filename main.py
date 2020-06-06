'''
filename: main.py
content: main function

提示: 总main函数可能会运行1h以上，建议分段运行

#TODO
针对各个部分进行多进程优化从而减少反复调用poolFit的进程开销
'''

import numpy as np
import visualization as vis
from preprocess import dataprocess
from networks import NeuralNetworkClassifier
from concurrent.futures import ProcessPoolExecutor
from tools import array2Label, train_test_split, accuracy_score, ttest

filepath = "datapca_10.csv"
X, y = dataprocess(filepath)

def oneFit(X, y, activation="relu", hidden_layers=(20, 20), test_size=0.2, loss=False):
    '''
    process of one fit
    loss: return loss during iteration information if True
    return accuracy or array of loss
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    y_label = array2Label(y_test)  # transform 2d array into 1d labels
    # 激活函数使用relu
    nn = NeuralNetworkClassifier(hidden_layer_sizes=hidden_layers, activation=activation)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    # random guess 0.006
    if not loss:
        return accuracy_score(y_label, y_pred)
    return nn.getIterLoss()

def multiFit(X, y, n=20 , activation="relu", hidden_layers=(20, 20), test_size=0.2):
    '''
    fit the model for n times and return mean accuracy and standard error
    INPUT
    X, y
    n: number of iterations
    activation: activation function
    hidden_layers: layer information for the model
    test_size
    RETURNS
    mean, std
    '''
    accuracy = np.zeros(n)
    for i in range(n):
        accuracy[i] = oneFit(X, y, activation=activation, 
                             hidden_layers=hidden_layers, 
                             test_size=test_size)
    mean = np.mean(accuracy)
    std = np.std(accuracy, ddof=1)
    return mean, std

def poolFit(X, y, n_process=10, n=20, activation="relu", hidden_layers=(20, 20), test_size=0.2):
    '''
    multiFit with multi process
    max_worker: max number of process
    '''
    tasks = []
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for _ in range(n):
            tasks.append(pool.submit(oneFit, X, y, activation, 
                                     hidden_layers, test_size, False))
    accuracy = np.array([task.result() for task in tasks])
    mean = np.mean(accuracy)
    std = np.std(accuracy, ddof=1)
    return mean, std

def layerEffect(max_layers=5, max_nodes=40, min_layers=1, min_nodes=1 ,activation="relu"):
    '''
    test the effect of different hidden layer sizes
    including the number of layers and the number of nodes in each layer
    using relu as activation function 
    RETURNS
    mean, std: 2D arrays, array[i][j] holds for i+1 layers and j+1 nodes
    '''
    accuracy_mean = np.zeros((max_layers, max_nodes))
    accuracy_std = np.zeros((max_layers, max_nodes))
    for n_layer in range(min_layers-1, max_layers):
        temp_mean = np.zeros(max_nodes)
        temp_std = np.zeros(max_nodes)
        for n_node in range(min_nodes-1, max_nodes): 
            layers = [n_node+1 for i in range(n_layer+1)]
            mean, std = multiFit(X, y, hidden_layers=layers, activation=activation)
            temp_mean[n_node] = mean
            temp_std[n_node] = std
            # output
            print("layer num:", n_layer+1,"node num:", n_node+1, "--trained")
        accuracy_mean[n_layer] = temp_mean
        accuracy_std[n_layer] = temp_std
    return accuracy_mean, accuracy_std


def poolLayerEffect(n_process=10, max_layers=5, max_nodes=40, min_layers=1, min_nodes=1, activation="relu"):
    '''
    accelarate with poolFit
    '''
    accuracy_mean = np.zeros((max_layers, max_nodes))
    accuracy_std = np.zeros((max_layers, max_nodes))
    for n_layer in range(min_layers-1, max_layers):
        temp_mean = np.zeros(max_nodes)
        temp_std = np.zeros(max_nodes)
        for n_node in range(min_nodes-1, max_nodes):
            layers = [n_node+1 for i in range(n_layer+1)]
            mean, std = poolFit(X, y, n_process=n_process, 
                                hidden_layers=layers, 
                                activation=activation)
            temp_mean[n_node] = mean
            temp_std[n_node] = std
            # output
            print("layer num:", n_layer+1, "node num:", n_node+1, "--trained")
        accuracy_mean[n_layer] = temp_mean
        accuracy_std[n_layer] = temp_std
    return accuracy_mean, accuracy_std

def poolNodeEffect(max_nodes = 90, n_process=10):
    ''' 
    accelarate with multi process
    for one layer networks
    test how number of nodes affect accuracy among different activations
    RETURNS
    mean: an array of accuracy of (relu, sigmoid, tanh)
    mean [[relu],[sigmoid],[tanh]]
    '''
    mean = np.zeros((3, max_nodes))
    activations = ["relu", "sigmoid", "tanh"]
    for i in range(3):
        temp = poolLayerEffect(n_process=n_process,
                               max_layers=1, 
                               max_nodes=max_nodes, 
                               activation=activations[i])
        mean[i] = temp[0][0]
    return mean

def testSizeEffect(testSizes):
    '''
    test effect of different test sizes when splitting the data set
    '''
    n = len(testSizes)
    accuracy_mean = np.zeros(n)
    accuracy_std = np.zeros(n)
    for i in range(n):
        mean, std = multiFit(X, y, test_size=testSizes[i])
        accuracy_mean[i] = mean
        accuracy_std[i] = std
        print("Test size", testSizes[i], "--trained")
    return accuracy_mean, accuracy_std
        
def activationEffect():
    '''
    test difference between activation functions
    RETURNS
    losses array of (relu, sigmoid, tanh)
    '''
    # relu
    losses = []
    losses.append(oneFit(X, y, activation="relu", hidden_layers=(20, 20), loss=True))
    losses.append(oneFit(X, y, activation="sigmoid", hidden_layers=(80, ), loss=True))
    losses.append(oneFit(X, y, activation="tanh", hidden_layers=(50, ), loss=True))
    return losses

def main():
    # a glimpse
    # print(multiFit(X, y, activation="sigmoid", hidden_layers=(100, )))
    # print(poolFit(X, y, activation="sigmoid", hidden_layers=(100, )))

    # effect of # of layers and nodes with relu
    result_layer = layerEffect()
    vis.layers3D(result_layer[0], save=False)  # True to save

    # effect of # of layers and nodes with sigmoid
    # result_layer = layerEffect(
    #     max_layers=4, max_nodes=40, activation="sigmoid")
    # vis.layers3D(result_layer[0])

    # effect of # of layers and nodes with tanh
    # result_layer = layerEffect(
    #     max_layers=4, max_nodes=60, activation="tanh")
    # vis.layers3D(result_layer[0])

    # node effects of different activation functions for one layer
    # number of process 10  should change according to the machine
    # mean = poolNodeEffect(max_nodes=90, n_process=10)
    # vis.nodeLine(mean)

    # effect of testSizes
    # testSizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    # mean, std = testSizeEffect(testSizes)
    # vis.testSizeBar(mean, std, testSizes=testSizes)
    # for i in range(len(mean)-1):
    #     ttest(mean[i], mean[i+1], std[i], std[i+1], 20, 20)

    # loss change of diffent activation function 
    # losses = activationEffect()
    # vis.iterLossLine(losses)

if __name__ == "__main__":
    main()
