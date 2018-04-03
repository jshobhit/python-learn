# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:02:06 2017

@author: Shobhit
"""
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

# CNN Layers of Keras
from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
np.random.seed(7)
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.utils import np_utils

#from keras.utils import np_utils

def getPixel(row, column):
     img = str("p_"+str(row)+"_"+str(column))
     return img
 
def getImg():
    #For Training Data
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    #For Test Data
    #path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
    df = pd.read_csv(path)
    img = []
    for i in range(16):
        for j in range(8):
            x = getPixel(i, j)
            img.append(df[x])
    #image = np.array(img)
    return img

def loadData():
    #For Training Data
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    #For Test Data
    #path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
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
        
    labels= (df['Prediction']).tolist() 
    categories = {v:k for k,v in enumerate(sorted(list(set(labels))))}
    labels = np.array([categories[label] for label in labels])
#    labels = np.array(onehot(labels, 13))
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=0)
    return (X_train, y_train), (X_test, y_test)


###############################################################################################


(X_train, Y_train), (X_test, Y_test) = loadData()

#from matplotlib import pyplot as plt
#plt.imshow(tdataset[2])
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

X_train = X_train.reshape(X_train.shape[0],1,8,16).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,8,16).astype('float32')

#Y_train = Y_train.reshape(Y_train.shape[0],1,26)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 26)
Y_test = np_utils.to_categorical(Y_test, 26)

model = Sequential()
model.add(Convolution2D(24, (2, 1), activation = 'relu', input_shape=(16,8,1), data_format = 'channels_last'))
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Dropout(0.5))
model.add(Convolution2D(20, (2,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,1)))
#model.add(Dropout(0.45))

#model.add(Convolution2D(64, (2,4), activation = 'tanh'))
#model.add(MaxPooling2D(pool_size=(1,1)))
#model.add(Dropout(0.4))

#model.add(Convolution2D(32, (1,1), activation = 'tanh'))
#model.add(MaxPooling2D(pool_size=(2,1)))
#model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
#model.add(Dense(80, activation = 'relu'))
#model.add(Dense(80, activation = 'relu'))
#model.add(Dense(50, activation = 'relu'))

model.add(Dense(26, activation = 'softmax'))

#adam = optimizers.Adam(lr=0.00001, decay=0.0)
#sgd = optimizers.SGD(lr = 0.001, momentum = 0.99, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size = 32, nb_epoch = 3, verbose = 1)
model.fit(xtrain, ytrain, batch_size = 15, epochs = 10, validation_data=(xtest, ytst), verbose = 1)

scores = model.evaluate(xtest, ytst, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

###############################################################################################
model.save("C://Users//Shobhit//Desktop//weights8718.h5")

tpred = model.predict(tdataset)
predictions = []
for i in range(len(tpred)):
    predictions.append(charpred(i))

def pred(imgid):
    for i in range(26):
        if(tpred[imgid][i]==max(tpred[imgid])):
            return i
        
def charpred(imgid):
    for key, value in categories.items():
        if value == pred(imgid):
            return key



idlist = [i for i in range(len(predictions))]
dataframe = pd.DataFrame(data={"ID": idlist, "Prediction": predictions})
dataframe.to_csv("C://Users//Shobhit//Desktop//Prediction1.csv", sep = ',', index = False)
