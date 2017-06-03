import os
import numpy as np
from scipy.interpolate import spline
import pandas as pd
import json
import sklearn
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import skimage.io
import skimage.transform
import skimage.color
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, AveragePooling2D, GlobalAveragePooling2D, concatenate, Input, Dropout
from keras.optimizers import SGD, Adadelta, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, History


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
]


class FlippedImageDataGenerator(ImageDataGenerator):
    def __init__(self, flip_indices):
        super(FlippedImageDataGenerator, self).__init__()
        self.flip_indices = flip_indices

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


# def visualize_loss(history):
#     plt.plot(history.history['loss'], linewidth=3, label='train')
#     plt.plot(history.history['val_loss'], linewidth=3, label='valid')
#     plt.grid()
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.ylim(1e-3, 1e-2)
#     plt.yscale('log')
#     plt.show()


def visualize_loss():
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']
    for i in range(6):
        with open('model_specialists/history_%d.json' % i) as f:
            history = json.load(f)
            # loss = history['loss']
            # val_loss = history['val_loss']
            # xnew = np.linspace(0, 399, 100000)
            # loss_smooth = spline(np.arange(400), loss, xnew)
            # val_loss_smooth = spline(np.arange(400), val_loss, xnew)
            interval = 5
            x = list(range(0, 400, interval))
            loss = history['loss'][::interval]
            val_loss = history['val_loss'][::interval]
            plt.plot(x, loss, ':', color=colors[i])
            plt.plot(x, val_loss, label='group %d' % i, color=colors[i])
            print('|%f|%f|%f|%f|' % (
                history['loss'][-1], history['acc'][-1], history['val_loss'][-1], history['val_acc'][-1]))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()


def load_specialists(idx, test=False):
    assert 0 <= idx < 6
    if not test:
        df = pd.read_csv('training.csv')
    else:
        df = pd.read_csv('test.csv')

    df['Image'] = df['Image'].apply(lambda img_str: np.fromstring(img_str, sep=' '))
    config = SPECIALIST_SETTINGS[idx]
    df2 = df[list(config['columns']) + ['Image']].copy()
    del df
    df2 = df2.dropna()
    x = np.vstack(df2['Image'].values) / 255
    x = x.astype(np.float32)
    y = None
    if not test:
        y = df2[df2.columns[:-1]].values
        y = (y - 48) / 48
        y = y.astype(np.float32)
        x, y = sklearn.utils.shuffle(x, y, random_state=42)
    return x, y


def load(file_name, test=False):
    df = pd.read_csv(file_name)

    # drop rows with missing columns (7049 -> 2140)
    df = df.dropna()

    df['Image'] = df['Image'].apply(lambda img_str: np.fromstring(img_str, sep=' '))

    X = np.vstack(df['Image'].values) / 255
    X = X.astype(np.float32)

    y = None
    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        y = y.astype(np.float32)
        X, y = sklearn.utils.shuffle(X, y, random_state=42)

    # X.shape == (2140, 9216)
    # y.shape == (2140, 30)
    return X, y


def load2d(file_name, test=False):
    X, y = load(file_name, test)
    X = X.reshape(-1, 96, 96, 1)
    # X.shape == (2140, 96, 96, 1)
    return X, y


def load2d_specialists(idx, test=False):
    X, y = load_specialists(idx, test)
    X = X.reshape(-1, 96, 96, 1)
    return X, y

def model_cnn(classes=30):
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

    model_cnn.add(Flatten()) # Flatten the multi-dimentional input into 1-D for Dense layer
    model_cnn.add(Dense(500))
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(500))
    model_cnn.add(Activation('relu'))
    model_cnn.add(Dense(classes))

    return model_cnn


def run_specialists():
    if not os.path.exists('model_specialists'):
        os.mkdir('model_specialists')
    for i in range(6):
        model_path = 'model_specialists/model_%d.h5' % i
        X, y = load2d_specialists(i)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        flipgen = FlippedImageDataGenerator(SPECIALIST_SETTINGS[i]['flip_indices'])
        optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
        # optimizer = Adam(lr=0.001)
        model = model_cnn(len(SPECIALIST_SETTINGS[i]['columns']))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])
        early_stopping = EarlyStopping()
        checkpoint = ModelCheckpoint(model_path, period=10)
        history = History()
        model.fit_generator(
            flipgen.flow(X_train, y_train),
            samples_per_epoch=X_train.shape[0],
            epochs=400,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, history])
        model.save(model_path)
        with open('model_specialists/history_%d.json' % i, 'w') as f:
            hist = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'acc': history.history['acc'],
                'val_acc': history.history['val_acc']
            }
            json.dump(hist, f)


def generate_submission_csv():
    X, _ = load2d('test.csv', test=True)
    y_ = []
    for i in range(6):
        model = load_model('model_specialists/model_%d.h5' % i)
        y_.append(model.predict(X))
    y_ = np.hstack(y_)
    y_ = y_ * 48 + 48
    i = 0
    feature_dict = {}
    for config in SPECIALIST_SETTINGS:
        for col_name in config['columns']:
            feature_dict[col_name] = i
            i += 1
    df = pd.read_csv('IdLookupTable.csv')
    for i, row in df.iterrows():
        df.loc[i, 'Location'] = y_[row['ImageId'] - 1][feature_dict[row['FeatureName']]]
    df.drop(['ImageId', 'FeatureName'], axis=1, inplace=True)
    submission_path = os.path.join(os.path.dirname('model_specialists/'), 'submission.csv')
    df.to_csv(submission_path, index=False)


def main():
    #run_specialists()
    #generate_submission_csv()
    visualize_loss()


if __name__ == '__main__':
    main()
