"""
kr_mnist_layers.py: Multiclass classification of MNIST digits dataset using a ANN & then a CNN
In this example, we use Keras layers for pre-processing the images, so our data-sourcing 
involves just loading the images & splitting into train/cross-val & test sets

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import random
import math
# reduce warnings overload from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
# np.set_printoptions(precision=6, linewidth=1024, suppress=True)
# plt.style.use('seaborn')
# sns.set_style('darkgrid')
# sns.set_context('notebook',font_scale=1.10)

# Keras imports
import tensorflow as tf
print(f"Using Tensorflow version: {tf.__version__}")
USING_TF2 = tf.__version__.startswith("2")
# we need version 2.2.0+
tf_ver_num = float(tf.__version__[:3])
assert tf_ver_num >= 2.2, f"FATAL: need Tensorflow version >= 2.2, got {tf_ver_num}"

# using Tensorflow's implementation of Keras
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Reshape, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Normalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
# my helper functions for Keras
import kr_helper_funcs as kru

seed = kru.seed_all()
kru.setupModeules()

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

# some globals
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 25, 64, 0.001, 0.0005
MODEL_SAVE_DIR = os.path.join('.', 'model_states')

# --------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------


def display_sample(sample_images, sample_labels, sample_predictions=None, num_rows=5, num_cols=10,
                   plot_title=None, fig_size=None):

    import seaborn as sns

    """ display a random selection of images & corresponding labels, optionally with predictions
        The display is laid out in a grid of num_rows x num_col cells
        If sample_predictions are provided, then each cell's title displays the prediction
        (if it matches actual) or actual/prediction if there is a mismatch
    """
    assert sample_images.shape[0] == num_rows * num_cols

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(num_rows, num_cols, figsize=((14, 10) if fig_size is None else fig_size),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35}, squeeze=True)

        for r in range(num_rows):
            for c in range(num_cols):
                image_index = r * num_cols + c
                ax[r, c].axis("off")
                # show selected image
                # NOTE: matplotlib expects images to be of shape (H,W) for grayscale images
                plt_image = sample_images[image_index]
                if plt_image.ndim > 2:
                    plt_image = np.squeeze(plt_image)  # drop the channels

                ax[r, c].imshow(plt_image, cmap="Greys", interpolation='nearest')

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title("%d" % sample_labels[image_index])
                else:
                    # else check if prediction matches actual value
                    true_label = sample_labels[image_index]
                    pred_label = sample_predictions[image_index]
                    prediction_matches_true = (true_label == pred_label)
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = "%d" % true_label
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = '%d/%d' % (true_label, pred_label)
                        title_color = 'r'
                    # display cell title
                    title = ax[r, c].set_title(title)
                    plt.setp(title, color=title_color)
        # set plot title, if one specified
        if plot_title is not None:
            f.suptitle(plot_title)

        plt.show()
        plt.close()


def load_data(debug=True):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    """
    loads the MNIST dataset from keras.datasets.mnist package
      - splits the images & labels into train/cross-validation & test sets
    NOTE: all other pre-processing is done in the layers of the model itself!
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if debug:
        print('Before preprocessing:')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))

    # Per Andrew Ng's advise from his Structured ML course:
    # the test & cross-validation datasets must come from the same distribution
    # So, we are going to split X_test/y_test into cross-val & test datasets
    # (we'll use the fact that X_test/y_test is a labelled dataset to our advantage)
    # We have 10,000 samples in X_test/y_test - we'll assign 8,000 to X_val/y_val & 2,000
    # examples to X_test/y_test (test dataset). This also gives us more records in our
    # training set :).

    # shuffle the training set (get rid of any implicit sorting)
    indexes = np.arange(X_train.shape[0])
    indexes = np.random.permutation(indexes)
    X_train = X_train[indexes]
    y_train = y_train[indexes]

    # shuffle the test set (get rid of any implicit sorting)
    indexes = np.arange(X_test.shape[0])
    for _ in range(5):
        indexes = np.random.permutation(indexes)  # shuffle 5 times!
    X_test = X_test[indexes]
    y_test = y_test[indexes]
    # val_count = int(0.20 * X_train.shape[0])
    val_count = 8000
    X_val = X_test[:val_count]
    y_val = y_test[:val_count]
    X_test = X_test[val_count:]
    y_test = y_test[val_count:]

    # NOTE: No further pre-processing is done in this function
    # Normally I would convert images to float32 type & normalize data
    # all this pre-processing is done in the model's layers

    if debug:
        print('After preprocessing:')
        print(' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
        print(' - X_val.shape = {}, y_val.shape = {}'.format(X_val.shape, y_val.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def step_lr(epoch):
    global LEARNING_RATE

    initial_lr = LEARNING_RATE
    drop_rate = 0.5
    epochs_drop = 10.0
    new_lr = initial_lr * math.pow(drop_rate, math.floor((1 + epoch) / epochs_drop))
    return new_lr


def build_model_ann(l2_loss_lambda=None):
    # build the model using functional API + keras' new experimental
    # Rescaling layer, which pre=processes images in the model itself

    l2_reg = None if l2_loss_lambda is None else l2(l2_loss_lambda)

    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH))

    # pre-processing layers
    x = Rescaling(1.0 / 255.0)(inputs)
    x = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))(x)

    # flatten the inputs, so they can be fed to Dense layers
    x = Flatten()(x)

    # prediction layers
    x = Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.10)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.10)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=out)

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def build_model_cnn(l2_loss_lambda=None):
    l2_reg = None if l2_loss_lambda is None else l2(l2_loss_lambda)

    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH))

    # pre-processing layers
    x = Rescaling(1.0 / 255.0)(inputs)
    x = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))(x)

    # feature selection layers (Conv2D)
    x = Conv2D(128, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer=l2_reg)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='SAME', activation='relu', kernel_regularizer=l2_reg)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.10)(x)
    x = Flatten()(x)

    # prediction layers
    x = Dense(512, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.20)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


DO_TRAINING = True
DO_PREDICTION = True
SHOW_SAMPLE = True
SAMPLE_SIZE = 50
USE_CNN = True
MODEL_FILE_NAME = 'kr_mnist_cnn3' if USE_CNN else 'kr_mnist_dnn3'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILE_NAME)


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(f"X_train.shape = {X_train.shape} - y_train.shape = {y_train.shape} " +
          f"- X_val.shape = {X_val.shape} - y_val.shape = {y_val.shape} " +
          f"- X_test.shape = {X_test.shape} - y_test.shape = {y_test.shape}")

    if SHOW_SAMPLE:
        print(f"Displaying sample of {SAMPLE_SIZE} images...")
        rand_indexes = np.random.randint(0, len(X_test), SAMPLE_SIZE)
        sample_images = X_test[rand_indexes]
        sample_labels = y_test[rand_indexes]
        display_sample(sample_images, sample_labels, plot_title='Sample of %d images' % SAMPLE_SIZE)

    if DO_TRAINING:
        if USE_CNN:
            print('Using CNN architecture...')
            model = build_model_cnn(l2_loss_lambda=L2_REG)
            print(model.summary())
        else:
            print('Using ANN/MLP architecture...')
            model = build_model_ann(l2_loss_lambda=L2_REG)
            print(model.summary())

        lr_scheduler = LearningRateScheduler(step_lr)
        hist = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(X_val, y_val), callbacks=[lr_scheduler])
        kru.show_plots(hist.history, metric='sparse_categorical_accuracy')

        # evaluate model performance
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print(f'  Training dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_val, y_val)
        print(f'  Cross-val dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')
        loss, acc = model.evaluate(X_test, y_test)
        print(f'  Test dataset  -> loss: {loss:.4f} - acc: {acc:.4f}')

        kru.save_model(model, MODEL_SAVE_PATH)
        del model

    if DO_PREDICTION:
        model = kru.load_model(MODEL_SAVE_PATH)
        print(model.summary())

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print('Sample labels (50): ', y_test[:50])
        print('Sample predictions (50): ', y_pred[:50])
        print('We got %d/%d incorrect!' % ((y_pred != y_test).sum(), len(y_test)))

        if SHOW_SAMPLE:
            # display sample predictions
            rand_indexes = np.random.randint(0, len(X_test), SAMPLE_SIZE)
            sample_images = X_test[rand_indexes]
            sample_labels = y_test[rand_indexes]
            sample_predictions = y_pred[rand_indexes]

            model_type = 'CNN' if USE_CNN else 'ANN'
            display_sample(sample_images, sample_labels, sample_predictions,
                           num_rows=5, num_cols=10, plot_title=f'Keras {model_type} - {SAMPLE_SIZE} random predictions')

            del model

# --------------------------------------------------------------
# Results:
#     MLP/ANN:
#       - Training acc -> 99.52
#       - Cross-val acc -> 98.29
#       - Testing acc -> 98.29
#     CNN:
#       - Training acc -> 99.58
#       - Cross-val acc -> 99.27
#       - Testing acc -> 99.15
# CNN is a bit better than the ANN/MLP
# --------------------------------------------------------------


if __name__ == "__main__":
    main()
