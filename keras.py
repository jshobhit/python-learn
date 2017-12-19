# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:02:06 2017

@author: Shobhit
"""
# Simple 2 layer Dense Network.

from keras.models import Sequential

# Core Layers of Keras
from keras.layers import Dense, Dropout, Activation, Flatten

# CNN Layers of Keras
from keras.layers import Convolution2D, MaxPooling2D

import numpy as np

np.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model1 = Sequential()

model1.add(Dense(12, input_dim=8, activation='relu'))
model1.add(Dense(8, activation = 'relu'))
model1.add(Dense(1, activation = 'sigmoid'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model1.fit(X, Y, epochs = 150, batch_size = 10)
scores = model1.evaluate(X, Y)
print("\n %s : %.2f%%" % (model1.metrics_names[1], scores[1]*100))

# Convolutional NN with MNIST

from keras.utils import np_utils
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from matplotlib import pyplot as plt
plt.imshow(X_train[0])
X_train = X_train.reshape(X_train.shape[0],1,28,28)
X_test = X_test.reshape(X_test.shape[0],1,28,28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape=(1,28,28), data_format = 'channels_first'))

model.add(Convolution2D( 32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size = 32, nb_epoch = 10, verbose = 1)

score = model.evaluate(X_test, Y_test, verbose=0)