'''
filename: tools.py
content: some useful tools
'''
import numpy as np
from scipy import sparse
from random import seed, choices

def gen_batches(n, batch_size, min_batch_size=0):
    '''
    Generator to create slices containing batch_size elements
    n : num of batches
    batch_size : Number of element in each batch
    min_batch_size : Minimum batch size to produce

    yield slice
    '''
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)

def accuracy_score(y_true, y_pred, normalize=True):
    '''
    accuracy score
    normalize
        True: return proportion
        False: return number of accurate predict
    '''
    score = 0
    if(len(y_true)!=len(y_pred)):
        raise ValueError("y_true and y_pred have different length")
    for i in range(len(y_true)):
        if y_pred[i]==y_true[i]:
            score += 1
    if normalize:
        return score/len(y_pred)
    else:
        return score


def sparse_dot(a, b):
    '''
    dot product for sparse matrix
    '''
    if a.ndim>2 or b.ndim>2:
        if sparse.issparse(a):
            # dim(b)>2
            # 滚动轴
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            # 矩阵乘法
            ret = a@b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # dim(a)>2
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d@b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a@b
    return ret

def train_test_split(X, y, test_size=0.3):
    '''
    split X y into training set and test set
    return
        X_train, X_test, y_train, y_test
    '''
    if y.ndim == 1:
        y = np.reshape(y, (-1,1))
    n_features = X.shape[1]
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_test = int(np.floor(test_size*len(y)))
    X_test, y_test = data[:n_test, :n_features], data[:n_test, n_features:]
    X_train, y_train = data[n_test:, :n_features], data[n_test:, n_features:]
    return X_train, X_test, y_train, y_test

def array2Label(y):
    '''
    transform a 2d array of multi output y into a 1d array
    for example y = [[0,1,0],[0,0,1],[1,0,0]]
    y_label = [1, 2, 0]  represents the index of 1 in the row
    already assure y only has one nonzero value for each row
    '''
    row, col = np.nonzero(y)
    return col

class KFold:
    def __init__(self):
        #TODO
        pass

    def split(self, X):
        pass

if __name__ == "__main__":
    # X = np.array([np.arange(i,i+5) for i in range(100)])
    # y = np.arange(100)
    # print(X.ndim)
    # print(y.ndim)
    # X1, X2, y1, y2 = train_test_split(X, y, test_size=0.3)
    # print(len(X1))
    # print(len(X2))
    # print(y1)
    # print(y2)
    y_ = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,1,0,0],[1,0,0,0,0]])
    print("test11")
    print(array2Label(y_))
