import numpy as np


def softmax(X):
    X = np.exp(X)
    X /= np.sum(X)
    return X

X = np.array([1,2,3,4,5,6])
# print(np.exp(X, out=X))
print(softmax(X))

def expit(X):
    return 1/(1+np.exp(-X))

print(expit(X))
b = np.array([3,4,5])

print(X.mean()/2)
print(np.mean(X)/2)
X = X.reshape((-1,2))
print(X)
print(X.shape)