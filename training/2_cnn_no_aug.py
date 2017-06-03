import os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from collections import OrderedDict
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD 
import pickle

## Load the dataset in 1-D form
def load(file_name, test=False, cols=None):
    df = pd.read_csv(file_name)

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols: # choose given row
        df = df[list(cols) + ['Image']]

    # print(df.count()) Basic count of different features
    df = df.dropna() # We can see that some values are missing, we just use those completed data

    X = np.vstack(df['Image'].values) / 255. # Normalization
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

# Load the dataset in 2-D form
def load2d(file_name, test=False, cols=None):
    X, y = load(file_name, test, cols)
    X = X.reshape(-1, 96, 96, 1) # if theano backend --> -1 , 1, 96, 96
    return X, y

if __name__ == "__main__":

    # We need to build model: cnn_100 / cnn_200 / cnn_300 / cnn_400

    print("CNN")
    nb_epoch = int(input("nb_epoch = "))
    file_path = input("Save as (e.g. cnn_100.h5) : ")
    X2d, y2d = load2d('training.csv')

    model_cnn = Sequential()
    
    model_cnn.add(Convolution2D(32, 3, 3, input_shape=(96, 96, 1)))
    model_cnn.add(Activation('relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Convolution2D(64, 2, 2))
    model_cnn.add(Activation('relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Convolution2D(128, 2, 2))
    model_cnn.add(Activation('relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Flatten())
    model_cnn.add(Dense(500))
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dense(500))
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dense(30))
    
    sgd = SGD(lr=0.01, momentum = 0.9, nesterov = True)
    model_cnn.compile(loss = 'mean_squared_error', optimizer = sgd, metrics=['acc'])
    
    hist = model_cnn.fit(X2d, y2d, nb_epoch = nb_epoch, batch_size = 1, validation_split = 0.2)
    with open(file_path + '.history','w') as f:
        f.write(str(hist.history))
    # Save model
    model_cnn.save(file_path)  


