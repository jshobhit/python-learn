from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Based on the implementation here : https://github.com/Lasagne/Recipes/blob/master/papers/densenet/densenet_fast.py
'''

def conv_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout

    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with batch_norm, relu and convolution2d added

    '''

    x = Activation('relu')(ip)
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D

    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(nb_filter, 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones

    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def create_dense_net(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):
    ''' Build the create_dense_net model

    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay

    Returns: keras tensor with nb_layers of conv_block appended

    '''

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)

    densenet = Model(input=model_input, output=x, name="create_dense_net")

    if verbose: print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

def genOutput():
    testpath = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//test.csv"
    testdf = pd.read_csv(testpath)    
    
    data= np.asarray(testdf.drop(['Prediction', 'Id', 'NextId','Position','SNR'], axis = 1))
    #dataset = data.reshape(data.shape[0],16, 8)
    data = data.reshape(data.shape[0], 16, 8, 1)    
    y = model.predict(data)
        
    pred = [labels[np.argmax(y[i])] for i in range(len(data))]
    final = pd.DataFrame(data={"Id":[i for i in range(len(pred))],"Prediction":pred})
    final.to_csv("C://Users//Shobhit//Desktop//dense1.csv", sep = ',', index = False)


if __name__ == "__main__":
    path = "C://Users//Shobhit//Desktop//Current Shit To-Do//Theory and Algorithms for Machine Learning//train.csv"
    df = pd.read_csv(path)
        
    Y = pd.get_dummies(df['Prediction'])
    Y1 = df['Prediction']
    labels = Y.columns.values.tolist()
        
    xtrain, xtest, ytrain, ytest =  train_test_split(df, Y, test_size = 0.2, random_state = 10)
    
    train = np.asarray(xtrain.drop(['Prediction', 'Id', 'NextId','Position','SNR','Nextchar'], axis = 1))
    test = np.asarray(xtest.drop(['Prediction', 'Id','NextId', 'Position', 'SNR','Nextchar'], axis = 1))
    
    xtrain = train.reshape(train.shape[0], 16, 8, 1).astype('float32')
    xtest = test.reshape(test.shape[0], 16, 8, 1).astype('float32')
    
    ytr = np.asarray(ytrain)
    ytst = np.asarray(ytest)
    
    imgdim = (16, 8, 1)
    classes = 26
    
    model = create_dense_net(nb_classes = classes, img_dim = imgdim, depth=40, growth_rate=40, dropout_rate = 0.4)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        
    #model.fit(xtrain, ytrain, batch_size = 15, epochs = 10, validation_data=(xtest, ytst), verbose = 1)
    
    model.load_weights("C://Users//Shobhit//Desktop//Current Shit To-Do//weights12.h5")
    
    scores = model.evaluate(xtest, ytst, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
    ypred = model.predict(xtest)
    
    genOutput()                

