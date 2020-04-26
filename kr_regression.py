"""
kr_regression.py: Univariate regression with Keras on synthesized data

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Tensorflow & Keras imports
import tensorflow as tf
print(f"Using Tensorflow version: {tf.__version__}")
assert tf.__version__.startswith("2"), "ERROR: this module requires Tensorflow 2.0!`"

# using Tensorflow's implementation of Keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# to ensure that you get consistent results across runs & machines
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# log only error from Tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# my helper functions for Keras
import kr_helper_funcs as kru

def generate_temp_data(m, c, min_val=1.0, max_val=50.0, numelems=100, std=10.0):
    tf.random.set_seed(123)

    X = tf.linspace(min_val, max_val + 1.0, numelems)
    X = tf.reshape(X, (-1,1))
    noise = tf.random.uniform((numelems,1), -(std-1), std, dtype=tf.dtypes.float32)
    y = m * X + c + noise
    return X.numpy(), y.numpy()

def build_model():
    K.clear_session()
    model = Sequential([
        Dense(units=1, input_shape=[1]),
    ])
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=[kru.r2_score])
    return model

def main():
    # generate data with noise
    M, C = 1.8, 32.0
    X, y = generate_temp_data(M, C, numelems=500, std=25)
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")

    # display plot of generated data
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, s=40, c='steelblue')
    plt.title(f'Original Data -> $y = {M:.2f} * X + {C:.2f}$')
    plt.show()

    model = build_model()
    print(model.summary())
    print('Before training: ')
    print(f'   Weight: {model.layers[0].get_weights()[0]} bias: {model.layers[0].get_weights()[1]}')

    # train the model
    hist = model.fit(X, y, epochs=5000, batch_size=32)
    kru.show_plots(hist.history, metric='r2_score', plot_title="Performance Metrics")

    print('After training: ')
    print(f'   Weight: {model.layers[0].get_weights()[0]} bias: {model.layers[0].get_weights()[1]}')



if __name__ == "__main__":
    main()
