# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 23:58:54 2018

@author: Shobhit
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D

np.random.seed(7)

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

#def loadData():
#    dataset = []
#    data = getImg()
#    for i in range(105):
#        print("Appending Image: "+str(i))
#        test = []
#        for j in range(128):
#            test.append(np.array(data[j][i]))
#            x = np.array(test)
#        dataset.append(x.reshape(16,8))


def loadData():
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
    df = pd.read_csv(path)
    nextid = df['NextId']
    pos = df['Position']
    x = []
    for i in range(16):
        for j in range(8):
            x.append(getPixel(i,j))
     
    dataset = []

    for i in range(128):
        dataset.append(df[x[i]])
    
    dataset.append(nextid)
    dataset.append(pos)
    data = np.asarray(dataset)
    dataset = np.transpose(data)    
    return np.transpose(data)

def getCategories():
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    df = pd.read_csv(path)    
    labels= (df['Prediction']).tolist() 
    categories = {v:k for k,v in enumerate(sorted(list(set(labels))))}
    return categories  

def show(imgid):
    from matplotlib import pyplot as plt
    plt.imshow(dataset[imgid])
    print(labels[np.argmax(y[imgid])])
    
def buildModel():
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential
    from keras.regularizers import l2
    import numpy as np
    
    # as described in http://yuhao.im/files/Zhang_CNNChar.pdf
    model = Sequential()
    model.add(Conv2D(64, (3, 3),activation='relu', padding='same', strides=(1, 1),input_shape=(1, 16, 8)))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3),activation='relu', padding='same', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    
    model.add(Conv2D(256, (3, 3),activation='relu', padding='same', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])



tdata = np.asarray(dataset)
tdata = np.expand_dims(tdata, axis = 2)
tdata = tdata.reshape(tdata.shape[0],1,16,8)
model.load_weights("C://Users//Shobhit//Desktop//weights.h5")

twindow = tdata[43], tdata[22], tdata[15]
twindow = np.asarray(twindow)
y = model.predict(tdata)

def pred(imgid):
    for i in range(26):
        if(y[imgid][i]==max(y[imgid])):
            return i

predictions = []
for i in range(len(y)):
    predictions.append(pred(i))

def show(imgid):
    from matplotlib import pyplot as plt
    plt.imshow(dataset[imgid])  
    print(pred[imgid])

def compare(imgid):
    show(imgid)
    #predict = df['Prediction']
    print("Predicted: "+str(charpred(predictions[imgid])))
    print("Actual: "+str(predict[imgid]))

def charpred(imgid):
    for key, value in categories.items():
        if value == imgid:
            return key

def getSNR(img):
    count = 0
    noise = 0
    for i in range(16):
        for j in range(8):
            if(img[i][j] == 1):
                count += 1
            else:
                noise += 1
    print(count/noise)
    
    
from sklearn.metrics import accuracy_score

accuracy_score(predictions, predict)
prediction_classses = model.predict_classes(tdata)