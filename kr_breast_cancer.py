"""
kr_breast_cancer.py: Binary classification of Wisconsin Breast Cancer dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use.
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pathlib

# reduce warnings overload from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
# using Tensorflow's implementation of Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# my helper functions for Keras
import kr_helper_funcs as kru

seed = kru.seed_all()
# tweaks for Numpy, Pandas, Matplotlib & seaborn
kru.setupSciLabModules()

# # tweaks for libraries
# np.set_printoptions(precision=6, linewidth=1024, suppress=True)
# plt.style.use('seaborn')
# sns.set_style('darkgrid')
# sns.set_context('notebook', font_scale=1.10)

# Keras imports
print(f"Using Tensorflow version: {tf.__version__}. " +
      f"GPU {'is available :)' if tf.test.is_gpu_available() else 'is not available :('}")
USING_TF2 = tf.__version__.startswith("2")

# # to ensure that you get consistent results across runs & machines
# seed = 123
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# if USING_TF2:
#     tf.random.set_seed(seed)
# else:
#     tf.compat.v1.set_random_seed(seed)
# # log only error from Tensorflow
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # reduce warnings overload from Tensorflow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data_file_path = pathlib.Path(".")
print(f"Current path {data_file_path}")
data_file_path = data_file_path / "data" / "wisconsin_breast_cancer.csv"
print(f"Data file path is: {data_file_path.absolute()}", flush=True)
assert os.path.exists(data_file_path.absolute()), f"{data_file_path.absolute()} - data file does not exist!"
#data_file_path = './data/wisconsin_breast_cancer.csv'


def download_data_file():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

    df_cols = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
        "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

    print('Downloading data from %s...' % url)
    wis_df = pd.read_csv(url, header=None, names=df_cols, index_col=0)
    wis_df.to_csv(data_file_path)


def load_data(test_split=0.20):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(data_file_path.absolute()):
        download_data_file()

    assert os.path.exists(data_file_path.absolute()), "%s - unable to open file!" % data_file_path

    wis_df = pd.read_csv(data_file_path, index_col=0)

    # diagnosis is the target col - char
    wis_df['diagnosis'] = wis_df['diagnosis'].map({'M': 1, 'B': 0})
    f_names = wis_df.columns[wis_df.columns != 'diagnosis']

    X = wis_df.drop(['diagnosis'], axis=1).values
    y = wis_df['diagnosis'].values

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)

    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return (X_train, y_train), (X_test, y_test)


MODEL_SAVE_NAME = '.\model_states\kr_wbc_ann.h5'

# Hyper-parameters
NUM_FEATURES = 30
NUM_CLASSES = 1
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def build_model(l2_loss_lambda=None):
    l2_reg = None if l2_loss_lambda is None else l2(l2_loss_lambda)

    model = Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2_reg,
              input_dim=(NUM_FEATURES)),
        # Dropout(0.01),
        Dense(32, activation='relu', kernel_regularizer=l2_reg),
        # Dropout(0.01),
        #Dense(8, activation='relu', kernel_regularizer=l2_reg),
        Dense(NUM_CLASSES, activation='sigmoid')
    ])
    adam = Adam(learning_rate=LEARNING_RATE, decay=0.005)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


DO_TRAINING = True
DO_TESTING = True
DO_PREDICTION = True
L2_REG = 0.01


def main():

    (X_train, y_train), (X_test, y_test) = load_data()
    print(f"X_train.shape = {X_train.shape} - y_train.shape = {y_train.shape} " +
          f"- X_test.shape = {X_test.shape} - y_test.shape = {y_test.shape}")
    y_train, y_test = y_train.astype(np.float), y_test.astype(np.float)

    if DO_TRAINING:
        model = build_model(l2_loss_lambda=L2_REG)
        print(model.summary())

        print('Training model...')
        hist = model.fit(X_train, y_train, validation_split=0.20,
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        kru.show_plots(hist.history, metric='accuracy')

        # evaluate model performance
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print(f'  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_test, y_test)
        print(f'  Test dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')

        kru.save_model(model, MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        model = kru.load_model(MODEL_SAVE_NAME)
        print(model.summary())

        y_pred = np.round(model.predict(X_test)).reshape(-1)
        # display output
        print('Sample labels: ', y_test)
        print('Sample predictions: ', y_pred)
        print('We got %d/%d correct!' %
              ((y_test == y_pred).sum(), len(y_test)))

# --------------------------------------------------
# Results:
#   Training set -> acc: 98.46%
#   Test set     -> acc: 98.25%
# --------------------------------------------------


if __name__ == "__main__":
    main()
