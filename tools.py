'''
filename: tools.py
content: some useful tools
'''
import numpy as np
from scipy import sparse

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

def train_test_split(X, y, test_size):
    #TODO
    pass

class KFold:
    def __init__(self):
        #TODO
        pass

    def split(self, X):
        pass
