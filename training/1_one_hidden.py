import os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from collections import OrderedDict
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation
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

if __name__ == "__main__":

    print("One hidden layer")
    file_path = raw_input("Save as (e.g. hidden_100.h5) : ")
    X1d, y1d = load('training.csv')
    model_hidden = Sequential() 
    model_hidden.add(Dense(100, input_dim = 9216)) # Input the 1-D data
    model_hidden.add(Activation("relu"))
    model_hidden.add(Dense(30))
    sgd = SGD(lr = 0.01, momentum = 0.9, nesterov = True)
    model_hidden.compile(loss = 'mean_squared_error', optimizer = sgd, metrics=['acc'])
    hist = model_hidden.fit(X1d, y1d, nb_epoch = 400, batch_size = 1, validation_split = 0.2)
    with open(file_path + '.history','w') as f:
        f.write(str(hist.history))

    # Save model
    model_hidden.save(file_path)  


