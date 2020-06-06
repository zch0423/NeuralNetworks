'''
filename: visualization.py
content:
    functions to do with visualization of the result
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def nodeLine(mean, save=False):
    '''
    line plot
    INPUT
    mean accuracy array like [[relu],[sigmoid],[tanh]]
    '''
    fig, ax = plt.subplots()
    x = np.arange(1, len(mean[0])+1, 1)
    for each, label in zip(mean, ["ReLU", "Sigmoid", "Tanh"]):
        ax.plot(x, each, label=label)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.title("Effect of Nodes")
    plt.legend()
    if save:
        plt.savefig("nodeEffect.png", dpi=600)
    plt.show()

def layers3D(mean, save=False, scatter=False):
    '''
    mean: array with shape of (n_layers, n_nodes)
    plot a 3D figure to show the accuracy towards the number of layers and nodes
    '''
    n_layers, n_nodes = mean.shape
    ax = plt.axes(projection='3d')
    if scatter:
        # accuracy mean
        z_data = mean.ravel()
        # nodes
        x_data = [node for node in range(1, n_nodes+1)]*n_layers
        # layers
        y_data = [layer for _ in range(n_nodes) for layer in range(1, n_layers+1)]
        ax.scatter3d(x_data, y_data, z_data, c=z_data, cmap=cm.Greens)
    else:
        x_data = np.arange(1, n_nodes+1, 1)
        y_data = np.arange(1, n_layers+1, 1)
        x_data, y_data = np.meshgrid(x_data, y_data)
        ax.plot_surface(x_data, y_data, mean, cmap=cm.Greens)
        # ax.bar3d(x_data, y_data, z_data, dx=1, dy=1/30, dz=z_data)

    plt.yticks([layer for layer in range(1, n_layers+1)])
    if save:
        plt.savefig("layerEffect3D.png", dpi=600)
    plt.show()

def testSizeBar(mean, std, testSizes, save=False):
    '''
    draw bar plot
    '''
    fig, ax = plt.subplots()
    n = len(testSizes)
    ind = np.arange(n)
    ax.bar(ind, mean, yerr=std)
    plt.xticks(ind, labels=[str(size) for size in testSizes])
    if save:
        plt.savefig("testSizeBar.png", dpi=600)
    plt.ylabel("Accuracy")
    plt.xlabel("Test Set Size")
    plt.title("Test Set Size Effect")
    plt.show()

def iterLossLine(losses, save=False):
    '''
    draw line of loss during iteration
    '''
    fig, ax = plt.subplots()
    for loss, label in zip(losses, ["ReLU", "Sigmoid", "Tanh"]):
        t = np.arange(1, len(loss)+1, 1)
        ax.plot(t, loss, label=label)
    plt.title("Loss Change")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig("iterLoss.png", dpi=600)
    plt.show()

#%%
if __name__ == "__main__":
    # test
    # mean = np.array([[j for j in range(30, 0, -1)] for i in range(5)])
    # layers3D(mean)

    # test size
    # testSizes = np.linspace(0.05, 0.35, 7)
    # accu = np.random.rand(7)
    # std = np.random.uniform(0, 0.2, size=7)
    # testSizeBar(accu, std, testSizes)
    # layers2D(np.random.rand(10))

    # losses = np.array([np.random.uniform(0.001, 0.03, 30+i*5) for i in range(3)])
    # iterLossLine(losses)

    mean = [[0.0094108,0.02806874,0.09574468,0.19091653,0.2903437,0.44967267,
      0.60040917, 0.73330606, 0.80466448 ,0.86513912],
     [0.00703764, 0.01080196, 0.02037643, 0.03256956, 0.04860884, 0.07635025,
        0.10490998, 0.13486088, 0.1710311,  0.2091653],
        [0.00826514, 0.01792144, 0.04026187, 0.06342062, 0.10474632, 0.14590835,
         0.1891162  ,0.23518822 ,0.28248773 ,0.3292144]]
    nodeLine(mean)
# %%
