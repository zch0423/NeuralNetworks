'''
filename: preprocess.py
content: preprocess data from pca data to required format for neural networks
'''
#%%
import numpy as np
import pandas as pd


def _nameToNumeric(y_names):
    '''
    y_names dataframe type
    transform y_names into y 
    y array of (n_samples, n_outputs)
    # y_label array of size of n_samples
    '''
    unique = y_names.unique()
    l1 = len(unique)
    l2 = len(y_names)
    # y_label = np.zeros(l2)
    y = np.zeros((l2, l1))
    for i in range(l2):
        ix = np.where(unique == y_names[i])[0][0]
        y[i][ix] = 1
        # y_label[i] = ix
    return y

def dataprocess(filepath):
    '''
    input:filepath of pca data
    output: array of X and y
    y
        shape=(n_samples, n_outputs)
        for example y = [[1,0,0],[0,0,1],[0,1,0]]
    # y_label
    #     index of y suggesting the class it belongs to
    #     for example y_label = [0, 2, 1] corresponding to y above
    '''
    data = pd.read_csv(filepath, index_col=0)
    data = data.reset_index(drop=True)
    y_names = data.iloc[:, 0]
    # y, y_label = _nameToNumeric(y_names)
    y = _nameToNumeric(y_names)
    X = data.iloc[:, 1:].to_numpy()
    # return X, y, y_label 
    return X, y

if __name__ == "__main__":
    filepath = "data360.csv"
    X, y = dataprocess(filepath)
    print(X)
    print("-"*20)
    for i in range(len(y)):
        # print(y[i], y_label[i])
        print(y[i])
