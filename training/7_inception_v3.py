# InceptionV3, Adam, 1000 epoches
import os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from collections import OrderedDict
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import pickle
from skimage import transform, color
from keras.applications.inception_v3 import InceptionV3

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

# Load the dataset in 2-D form and resize to inception-4v --> 299 x 299 x 3 
def load2d_inception(file_name, test=False, cols=None):
    X, y = load(file_name, test, cols) 
    X = X.reshape(-1, 96, 96)

    X_tran = []
    for img in X[:]:
        img = transform.resize(img, (299,299))
        img = color.gray2rgb(img)
        X_tran.append(img)
    return np.array(X_tran) , y

#####################
if __name__ == "__main__":

    file_path = input("Save as (e.g. hidden_100.h5) : ")

    X2d, y2d = load2d_inception('training.csv')
    model_pretrained = InceptionV3()
    X2d_features = np.array(model_pretrained.predict(X2d)) # 1000

    if X2d_features.shape[1] != 1000:
        print('Fuck' * 100)

    model_hidden = Sequential() 
    model_hidden.add(Dense(100, input_dim = 1000))
    model_hidden.add(Activation("relu"))
    model_hidden.add(Dense(30))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model_hidden.compile(loss = 'mean_squared_error', optimizer = adam, metrics=['acc'])
    hist = model_hidden.fit(X2d_features, y2d, nb_epoch = 1000, batch_size = 1, validation_split = 0.2)
    with open(file_path + '.history','w') as f:
        f.write(str(hist.history))

    # Save model
    model_hidden.save(file_path)  
