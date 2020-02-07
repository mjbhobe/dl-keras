"""
kr_iris.py: Multiclass classification of the Iris dataset

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Keras imports
import tensorflow as tf
print(f"Using Tensorflow version: {tf.__version__}")
USING_TF2 = tf.__version__.startswith("2")

# using Tensorflow's implementation of Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
# my helper functions for Keras
import kr_helper_funcs as kru

# to ensure that you get consistent results across runs & machines
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
if USING_TF2:
    tf.random.set_seed(seed)
else:
    tf.compat.v1.set_random_seed(seed)
# log only error from Tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_data(val_split=0.20, test_split=0.20):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.utils import to_categorical

    iris = load_iris()
    X, y, f_names = iris.data, iris.target, iris.feature_names

    # split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_split, random_state=seed)
    
    # standard scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # split into train/eval sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_split, random_state=seed)

    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size=test_split, random_state=seed)

    y_train = to_categorical(y_train, 4)
    y_val = to_categorical(y_val, 4)
    y_test = to_categorical(y_test, 4)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

MODEL_SAVE_NAME = 'kr_iris_ann'

# Hyper-parameters
NUM_FEATURES = 4
NUM_CLASSES = 4
NUM_EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.001
L2_REG = 0.005

        # model = WBCNet(NUM_FEATURES, 30, 30, NUM_CLASSES)
def build_model(inp_size, hidden1, hidden2, num_classes, l2_lambda=None):
    l2_reg = None if l2_lambda is None else l2(l2_lambda)

    model = Sequential([
        Dense(hidden1, activation='relu', input_dim=(inp_size), kernel_regularizer=l2_reg),
        Dense(hidden2, activation='relu', kernel_regularizer=l2_reg),
        Dense(num_classes, activation='softmax')
    ])
    #sgd = SGD(learning_rate=LEARNING_RATE, decay=0.005, nesterov=True, momentum=0.9)
    adam = Adam(learning_rate=LEARNING_RATE, decay=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model

DO_TRAINING = True
DO_PREDICTION = True

def main():

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(f"X_train.shape = {X_train.shape} - y_train.shape = {y_train.shape} " +
          f"- X_val.shape = {X_val.shape} - y_val.shape = {y_val.shape} " +
          f"- X_test.shape = {X_test.shape} - y_test.shape = {y_test.shape}")
    y_train, y_val, y_test = y_train.astype(np.float), y_val.astype(np.float), y_test.astype(np.float)

    if DO_TRAINING:
        model = build_model(NUM_FEATURES, 32, 32, NUM_CLASSES, L2_REG)
        print(model.summary())
        
        print('Training model...')
        hist = model.fit(X_train, y_train, validation_split=0.20, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        kru.show_plots(hist.history, metric='acc')

        # evaluate model performance
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print(f'  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_val, y_val)
        print(f'  Cross-val dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_test, y_test)
        print(f'  Test dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')

        kru.save_model(model, MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        model = kru.load_model(MODEL_SAVE_NAME)
        print(model.summary())
        
        y_pred = np.argmax(np.round(model.predict(X_test)), axis=1)
        y_true = np.argmax(y_test, axis=1)
        # display output
        print('Sample labels: ', y_true)
        print('Sample predictions: ', y_pred)
        print('We got %d/%d correct!' % ((y_true == y_pred).sum(), len(y_true)))

# --------------------------------------------------
# Results:
#   Training set -> acc: 96.88
#   Cross-val set -> acc: 94.74
#   Test set     -> acc: 100%
# --------------------------------------------------

if __name__ == "__main__":
    main()





