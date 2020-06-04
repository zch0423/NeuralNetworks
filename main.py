'''
filename: main.py
content: main function
'''
import numpy as np
from tools import array2Label, train_test_split, accuracy_score
from networks import NeuralNetworkClassifier
from preprocess import dataprocess

def main():
    filepath = "data360.csv"
    X, y = dataprocess(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_label = array2Label(y_test)  # transform 2d array into 1d labels
    # 激活函数使用relu
    nn = NeuralNetworkClassifier(hidden_layer_sizes=(300, 300), 
                                max_iter=1000, n_iter_no_change=10, tol=1e-5, activation="relu")
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    print(accuracy_score(y_label, y_pred))  #compared with 0.006
    #TODO
    '''
    some analysis about the effect of 
        number of layers 
        number of nodes for each layer
        different activation
        different pca size
    '''
main()