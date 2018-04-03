# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:16:34 2018

@author: Shobhit
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization

from keras.layers import Convolution1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.utils import np_utils


path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
df = pd.read_csv(path)

Y = pd.get_dummies(df['Prediction'])
labels = Y.columns.values.tolist()

xtrain, xtest, ytrain, ytest =  train_test_split(df, Y, test_size = 0.2, random_state = 10)

#xtrain = xtrain.drop(['Prediction', 'Id', 'NextId', 'Position'], axis = 1)
#xtest = xtest.drop(['Prediction', 'Id','NextId', 'Position'], axis = 1)

train = np.asarray(xtrain.drop(['Prediction', 'Id', 'NextId','Position','SNR'], axis = 1))
test = np.asarray(xtest.drop(['Prediction', 'Id','NextId', 'Position', 'SNR'], axis = 1))

train = np.expand_dims(train, axis = 2)
test = np.expand_dims(test, axis =2)

ytr = np.asarray(ytrain)
ytst = np.asarray(ytest)

#xtrain = train.reshape(train.shape[0], 8, 16)
xtrain = train.reshape(train.shape[0], 16, 8,1).astype('float32')
xtest = test.reshape(test.shape[0], 16, 8 ,1).astype('float32')
ytrain = train.reshape(train[0], 26, 1)
ytr = np.asarray(ytrain)
ytst = np.asarray(ytest)

ytr = np.expand_dims(ytr, axis = 2)
ytst = np.expand_dims(ytst, axis = 2)

ytrain = np.transpose(ytrain)

model = Sequential()
#model.add(Convolution1D(2, 2, activation = 'relu', input_shape=(130, 1)))
#model.add(MaxPooling1D(pool_size = 2))       
#model.add(Dropout(0.2))
model.add(Dense(132, input_dim = 130, activation = 'relu'))
#model.add(Convolution1D(2, 2, activation = 'relu'))
#model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.3))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
#model.add(Flatten())        
model.add(Dense(100, activation = 'tanh'))
model.add(Dense(80, activation = 'tanh'))
model.add(Dense(80, activation = 'tanh'))
model.add(Dense(80, activation = 'tanh'))
model.add(Dense(80, activation = 'tanh'))
model.add(Dense(65, activation = 'tanh'))
model.add(Dense(65, activation = 'tanh'))
#model.add(Flatten())    
model.add(Dense(26, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(train, ytr, validation_data=(test, ytst), epochs = 40, batch_size = 35)
model.save("C://Users//Shobhit//Desktop//weights.h5")

###############################################################################################
#################################    TEST YOUR MODEL ##########################################
###############################################################################################

model.evaluate(test, ytst, batch_size = 15)
y = model.predict_classes(test)

testdata = xtest.drop(['Position'], axis = 1)
testdata = np.asarray(testdata)
testdata = testdata.reshape(testdata.shape[0], 16, 8, 1)

labels = Y.columns.tolist()

def show(imgid):
    from matplotlib import pyplot as plt
    plt.imshow(testdata[imgid])

def compare(imgid):
    show(imgid)
#    predict = ytest['Prediction']
    print("Predicted: "+str(labels[y[imgid]]))
#    print("Actual: "+str(predict[imgid]))

print(model.predict_classes(np.asarray(testdata[:1])))

###############################################################################################
##################################    RESULTS TO CSV ##########################################
###############################################################################################

testpath = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
testdf = pd.read_csv(testpath)
testdata = testdf.drop(['Id','Prediction','NextId','Position', 'SNR'], axis = 1)
testdata = np.asarray(testdata)
testdata = testdata.reshape(testdata.shape[0], 16, 8, 1)
testdata = np.expand_dims(testdata, axis = 2)
ypred = model.predict_classes(testdata)

prediction = np.asarray(model.predict_classes(testdata))
#prediction = prediction.reshape(len(predictions))
prediction = ypred.tolist()

characters = []
for i in range(len(prediction)):
    characters.append(labels[ypred[i]])

imagedata = testdf.drop(['Id','Prediction','NextId','Position'], axis = 1)
imagedata = np.asarray(imagedata)
imagedata = imagedata.reshape(imagedata.shape[0], 16, 8)

def show(imgid):
    from matplotlib import pyplot as plt
    plt.imshow(imagedata[imgid])

def compare(imgid):
    show(imgid)
#    predict = ytest['Prediction']
    print("Predicted: "+str(labels[ypred[imgid]]))
#    print("Actual: "+str(predict[imgid]))

characters.tolist()    
prediction = pd.DataFrame(data={"Prediction": characters})
prediction.to_csv("C://Users//Shobhit//Desktop//8848.csv", sep = ',', index=True)


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################


def getSNR(img):
    count = 0
    noise = 0
    for i in range(128):
        if(img[i] == 1):
            count += 1
        else:
            noise += 1
    return (count/noise)

x = testdf
x = x.drop(['Prediction', 'Id', 'NextId', 'Position'], axis = 1)
x = np.asarray(x)
snr = []
for i in range(len(testdf)):
    snr.append(getSNR(x[i]))

snr = np.asarray(snr)
snr.tolist()
Snr = pd.DataFrame(data={"Prediction": snr})
Snr.to_csv("C://Users//Shobhit//Desktop//testsnr.csv", sep = ',', index=True)
