# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:40:05 2018

@author: Shobhit
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from keras.utils import np_utils

def getPixel(row, column):
     img = str("p_"+str(row)+"_"+str(column))
     return img
 
def getImg():
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    df = pd.read_csv(path)
    img = []
    for i in range(16):
        for j in range(8):
            x = getPixel(i, j)
            img.append(df[x])
    #image = np.array(img)
    return img

def loadData():
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
    df = pd.read_csv(path)
    dataset = []
    data = getImg()
    for i in range(41568):
        print("Appending Image: "+str(i))
        test = []
        for j in range(128):
            test.append(np.array(data[j][i]))
            x = np.array(test)
        dataset.append(x.reshape(16,8))
        
    labels= (df['Prediction']).tolist()s
    categories = {v:k for k,v in enumerate(sorted(list(set(labels))))}
    labels = np.array([categories[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=0)
    return (X_train, y_train), (X_test, y_test)
