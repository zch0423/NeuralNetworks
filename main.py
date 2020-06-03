'''
filename: main.py
content: main function
'''
# import numpy as np
# from tools import train_test_split, accuracy_score
# from networks import NeuralNetworkClassifier
# from preprocess import dataprocess

def main():
    filepath = "datapca_10.csv"
    X, y, y_label = dataprocess(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    nn = NeuralNetworkClassifier(hidden_layer_sizes=(100,))
    nn.fit(X_train, y_train)

#%%
#test 
import numpy as np
from tools import train_test_split, accuracy_score
from networks import NeuralNetworkClassifier
from preprocess import dataprocess

filepath = "datapca_10.csv"
X, y, y_label = dataprocess(filepath)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#TODO
# y_label也要对应split