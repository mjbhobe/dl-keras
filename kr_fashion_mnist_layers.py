"""
kr_fashion_mnist_layers.py: Multiclass classification of Fashion MNIST dataset using a CNN
In this example, we use Keras layers for pre-processing the images, so our data-sourcing 
involves just loading the images & splitting into train/cross-val & test sets

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import math
import random
import sys
import os
import kr_helper_funcs as kru
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Normalization
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.10)

# Keras imports
print(f"Using Tensorflow version: {tf.__version__}")
USING_TF2 = tf.__version__.startswith("2")
# we need version 2.2.0+
tf_ver_num = float(tf.__version__[:3])
assert tf_ver_num >= 2.2, f"FATAL: need Tensorflow version >= 2.2, got {tf_ver_num}"

# using Tensorflow's implementation of Keras
# my helper functions for Keras

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
# reduce Tensorflow warnings overload
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        If sample_predictions are provided, then each cell's title displays the prediction (if it matches
        actual)
        or actual/prediction if there is a mismatch
    """
    assert sample_images.shape[0] == num_rows * num_cols
    LABELS = {
        0: 'T-shirt/Top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot'
    }

    with sns.axes_style("whitegrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style(
            {"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

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
                    plt_image = np.squeeze(plt_image)  # drop the channels)
                ax[r, c].imshow(plt_image, cmap="Greys",
                                interpolation='nearest')

                if sample_predictions is None:
                    # show the actual labels in the cell title
                    title = ax[r, c].set_title(
                        f"{LABELS[sample_labels[image_index]]}")
                else:
                    # else check if prediction matches actual value
                    true_label = LABELS[sample_labels[image_index]]
                    pred_label = LABELS[sample_predictions[image_index]]
                    prediction_matches_true = (true_label == pred_label)
                    if prediction_matches_true:
                        # if actual == prediction, cell title is prediction shown in green font
                        title = f"{true_label}"
                        title_color = 'g'
                    else:
                        # if actual != prediction, cell title is actua/prediction in red font
                        title = f"{true_label}/{pred_label}"
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
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.utils import to_categorical

    """
    loads the Fashion MNIST dataset from keras.datasets.mnist package
      - splits the images & labels into train/cross-validation & test sets
    NOTE: all other pre-processing is done in the layers of the model itself!
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    if debug:
        print('Before preprocessing:')
        print(
            ' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
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

    val_count = 8000
    X_val = X_test[:val_count]
    y_val = y_test[:val_count]
    X_test = X_test[val_count:]
    y_test = y_test[val_count:]

    if debug:
        print('After preprocessing:')
        print(
            ' - X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
        print(' - X_val.shape = {}, y_val.shape = {}'.format(X_val.shape, y_val.shape))
        print(' - X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def step_lr(epoch):
    global LEARNING_RATE

    initial_lr = LEARNING_RATE
    drop_rate = 0.5
    epochs_drop = 10.0
    new_lr = initial_lr * \
        math.pow(drop_rate, math.floor((1+epoch)/epochs_drop))
    return new_lr


def build_model(l2_loss_lambda=None):
    l2_reg = None if l2_loss_lambda is None else l2(l2_loss_lambda)

    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH))

    # pre-processing layers
    x = Rescaling(1.0 / 255.0)(inputs)
    x = Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))(x)

    # feature selection layers (Conv2D)
    x = Conv2D(256, kernel_size=(3, 3), padding='SAME',
               activation='relu', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.30)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='SAME',
               activation='relu', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='SAME',
               activation='relu', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.20)(x)
    x = Flatten()(x)

    # prediction layers
    x = Dense(1024, activation='relu', kernel_regularizer=l2_reg)(x)
    x = Dropout(0.20)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(learning_rate=LEARNING_RATE, decay=0.005)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


DO_TRAINING = True
DO_PREDICTION = True
SHOW_SAMPLE = True
SAMPLE_SIZE = 50
MODEL_FILE_NAME = 'kr_fashion_mnist_layers'
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
        display_sample(sample_images, sample_labels,
                       plot_title='Sample of %d images' % SAMPLE_SIZE)

    if DO_TRAINING:
        model = build_model(l2_loss_lambda=L2_REG)
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
        print('We got %d/%d incorrect!' %
              ((y_pred != y_test).sum(), len(y_test)))

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
#    - Training acc -> 99.58
#    - Cross-val acc -> 99.27
#    - Testing acc -> 99.15
# --------------------------------------------------------------


if __name__ == "__main__":
    main()
