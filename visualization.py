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

def layers3D(mean, save=False, scatter=False):
    '''
    mean: array with shape of (n_layers, n_nodes)
    plot a 3D figure to show the accuracy towards the number of layers and nodes
    '''
    n_layers, n_nodes = mean.shape
    # accuracy mean
    z_data = mean.ravel()
    # nodes
    x_data = [node for node in range(1, n_nodes+1)]*n_layers
    # layers
    y_data = [layer for _ in range(n_nodes) for layer in range(1, n_layers+1)]
    ax = plt.axes(projection='3d')
    if scatter:
        ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap=cm.Greens)
    else:
        #TODO

    plt.yticks([layer for layer in range(1, n_layers+1)])
    if save:
        plt.savefig("layerEffect3D.png", dpi=600)
    plt.show()

#%%
if __name__ == "__main__":
    # test
    mean = np.random.rand(5, 30)
    layer3D(mean)



# %%
