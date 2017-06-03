# CNN using data augmentation

import os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD 
from keras.preprocessing.image import ImageDataGenerator
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

class FlippedImageDataGenerator(ImageDataGenerator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size / 2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )

        return X_batch, y_batch


if __name__ == "__main__":

    # We need to build model: cnn_100 / cnn_200 / cnn_300 / cnn_400

    print("CNN")
    nb_epoch = int(input("nb_epoch = "))
    file_path = input("Save as (e.g. cnn_100.h5) : ")
    X2d, y2d = load2d('training.csv')

    X_train, X_val, y_train, y_val = train_test_split(X2d, y2d, test_size = 0.2, random_state = 42)

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
    
    flipgen = FlippedImageDataGenerator()
    sgd = SGD(lr=0.01, momentum = 0.9, nesterov = True)
    model_cnn.compile(loss = 'mean_squared_error', optimizer = sgd, metrics=['acc'])
    hist = model_cnn.fit_generator(flipgen.flow(X_train, y_train), samples_per_epoch=X_train.shape[0],nb_epoch = nb_epoch ,validation_data=(X_val, y_val))
    with open(file_path + '.history','w') as f:
        f.write(str(hist.history))
    # Save model
    model_cnn.save(file_path)

