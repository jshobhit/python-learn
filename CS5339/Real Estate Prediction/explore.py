# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:13:19 2018

@author: Shobhit
"""

import pandas as pd
import numpy as np
import os
#import sklearn.feature_selection
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//pred2//"
hdbtrain = pd.read_csv(os.path.join(path,"hdb_train.csv"))
hdbtest = pd.read_csv(os.path.join(path, "hdb_test.csv"))

prtrain = pd.read_csv(os.path.join(path, "private_train.csv"))

y_hdb = hdbtrain['resale_price']
y_hdb = np.expand_dims(y_hdb, axis =0)
x_hdb = hdbtrain.drop(['resale_price'],axis =1)

cols = hdbnum.columns.values.tolist()
    # Column Names for HDB Train
    #['index', 0
    # 'block', 1
    # 'flat_model', 2
    # 'flat_type', 3
    # 'floor_area_sqm', 4
    # 'lease_commence_date', 5
    # 'month',6
    # 'resale_price',7
    # 'storey_range',8
    # 'street_name',9
    # 'town',10
    # 'latitude',11
    # 'longitude',12
    # 'postal_code',13
    # 'floor']14

hdbcategoric = hdbtrain.drop([cols[0], cols[4], cols[7], cols[14], cols[13], cols[12], cols[11]], axis = 1)

hdbnum = hdbtrain.drop([cols[0], cols[1], cols[2], cols[3], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10]], axis = 1)
hdbnum = np.asarray(hdbnum).astype('float32')

from keras.utils import np_utils
Xcat = []
for i in range(1):
    data = hdbcategoric[catcols[i]].values.tolist()
    di = dict(zip(data,np.arange(len(data))))
    Xcat[i] = (np_utils.to_categorical((hdbcategoric[catcols[i]].map(di)), len(set(hdbcategoric[catcols[i]])))) 
    del data, di
    
Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

for i in range(len(catcols)):
#    hdbcategoric[catcols[i]] = hdbcategoric[catcols[i]].astype('category') 
    hdbcategoric[catcols[i]] = hdbcategoric[catcols[i]].cat.codes
    

 model = Sequential()
 model.add(Dense(6, input_dim=8, init='normal', activation='relu'))
 model.add(Dense(10, init='normal', activation='relu'))
 model.add(Dense(20, init='normal', activation='relu'))
 model.add(Dense(10, init='normal', activation='relu'))
 model.add(Dense(1, init='normal'))
 model.compile(loss='mean_absolute_percentage_error', optimizer = 'adadelta')
model.fit(xtrain, ytrain, batch_size = 10, epochs  = 10, verbose = 1, validation_data = (xtest, ytest))

xtrain, xtest, ytrain, ytest = train_test_split(hdbcategoric, y_hdb, test_size = 0.2, random_state = 10)

hdbdata = []

for i in range(len(cols)):
    hdbdata.append(hdbnum[cols[i]])

for i in range(len(catcols)):
    hdbdata.append(hdbcategoric[catcols[i]])

data = pd.DataFrame(hdbdata, index=None)

lr = LinearRegression()
lr.fit(xtrain,ytrain)

out = model.predict(xtest)

metrics.mean_absolute_error(ytest, np.asarray(out))
metrics.r2_score(ytest, np.asarray(out))
