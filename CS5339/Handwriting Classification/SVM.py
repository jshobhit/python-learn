# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:03:37 2018

@author: Shobhit
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

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
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    df = pd.read_csv(path)
    nextid = df['NextId']
    pos = df['Position']
    
    dataset = []
    data = getImg()
    for i in range(len(nextid)):
        print("Appending Image: "+str(i))
        test = []
        for j in range(128):
            test.append(np.array(data[j][i]))
            x = np.array(test)
#        dataset.append(x.reshape(16,8))
        dataset.append(nextid[i])
        dataset.append(pos[i])
        dataset.append(x)

    
    labels= (df['Prediction']).tolist() 
    categories = {v:k for k,v in enumerate(sorted(list(set(labels))))}
    labels = np.array([categories[label] for label in labels])
#    labels = np.array(onehot(labels, 13))
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=0)
    return (X_train, y_train), (X_test, y_test)


def getCategories():
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    df = pd.read_csv(path)    
    labels= (df['Prediction']).tolist() 
    categories = {v:k for k,v in enumerate(sorted(list(set(labels))))}
    return categories  

def show(imgid):
    from matplotlib import pyplot as plt
    plt.imshow(dataset[imgid])

def compare(imgid):
    show(imgid)
    print("Predicted: "+str(charpred(prediction[imgid])))
    print("Actual: "+str(charpred(ytest[imgid])))
    
def charpred(imgid):
    for key, value in categories.items():
        if value == imgid:
            return key

from sklearn.metrics import accuracy_score

if __name__ =='__main__':
    (xtrain, ytrain), (xtest, ytest) = loadData()
    (xtrain, ytrain), (xtest, ytest) = (X_train, y_train), (X_test, y_test)
    xtrain = np.asarray(xtrain)
    xtest = np.asarray(xtest)
    xtrain = xtrain.reshape(len(xtrain), 130)
    xtest = xtest.reshape(len(xtest), 130)
    clf = svm.SVC()
    clf.fit(xtrain, ytr)
    
    tdataset = np.asarray(tdataset)
    tdataset = tdataset.reshape(len(tdataset), 128)    
    
    predictions = []
    for i in range(len(xtest)):
        predictions.append(clf.predict([xtest[i]]))

    
    prediction = np.asarray(predictions)
    prediction = prediction.reshape(len(predictions))
    prediction = prediction.tolist()
    
    characters = []
    for i in range(len(prediction)):
        characters.append(charpred(prediction[i]))
    
    characters.tolist()    
    prediction = pd.DataFrame(data={"Prediction": characters})
    prediction.to_csv("C://Users//Shobhit//Desktop//SVMpred.csv", sep = ',', index=True)
    
    accuracy_score(ytest, np.asarray(prediction))


##############################################################################


path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
df = pd.read_csv(path)
nextid = df['NextId'].tolist()
pos = df['Position'].tolist()
dataset = pd.DataFrame(data={"NextId":nextid, "Position":pos})
