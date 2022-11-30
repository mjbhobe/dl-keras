#!/usr/bin/env python
""" Fashion MNIST multiclass classification using Tensorflow 2.0 & Keras """
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                                     Input, MaxPooling2D, ELU, ReLU, Softmax)
from tensorflow.keras.models import load_model
import kr_helper_funcs as kru

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import kr_helper_funcs as kru

plt.style.use('seaborn')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Using Tensorflow {tf.__version__}")

EPOCHS, BATCH_SIZE, BUFFER_SIZE = 25, 64, 512


def load_fashion_data():
    """ load Fashion MNIST data & return datasets """
    from tensorflow.keras.datasets import fashion_mnist

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20,
                                                      random_state=SEED, stratify=y_train)
    # Normalize data.
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Reshape grayscale to include channel dimension.

    X_train = np.expand_dims(X_train, axis=3)
    X_val = np.expand_dims(X_val, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Process labels.
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_val = label_binarizer.fit_transform(y_val)
    y_test = label_binarizer.fit_transform(y_test)

    print(f"X_train.shape = {X_train.shape} - y_train.shape = {y_train.shape}\n"
          f"X_val.shape = {X_val.shape} - y_val.shape = {y_val.shape}\n"
          f"X_test.shape = {X_test.shape} - y_test.shape = {y_test.shape} ")

    train_ds = tf.data.Dataset.from_tensor_slices(X_train, y_train)
    val_ds = tf.data.Dataset.from_tensor_slices(X_val, y_val)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test, y_test)
    return train_ds, val_ds, test_ds


def build_model():
    input_layer = Input(shape=(28, 28, 1))

    x = Conv2D(filters=20, kernel_size=(5, 5), padding='same', strides=(1, 1))(input_layer)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=50, kernel_size=(5, 5), padding='same', strides=(1, 1))(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)

    x = Dense(units=500)(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=10)(x)
    output = Softmax()(x)

    model = Model(inputs=input_layer, outputs=output)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


MODEL_SAVE_PATH = os.path.join('./model_states', 'kr_fashion_mnist.h5')
DO_TRAINING = True
DO_PREDICTIONS = False


def main():
    # load & prepere the datasets for training
    print('Loading & preparing data...')
    train_dataset, val_dataset, test_dataset = load_fashion_data()
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)
    val_dataset = val_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)
    test_dataset = test_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)

    if DO_TRAINING:
        print('Training model...')
        # create the model
        model = build_model()
        print(model.summary())

        # train the model
        hist = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
        kru.show_plots(hist.history, metric='accuracy', plot_title='Fashion MNIST model performance')

        # evaluate model
        print('Evaluating model performance...')
        loss, acc = model.evaluate(train_dataset)
        print(f'  - Training data  -> loss = {loss:.3f} - acc = {acc:.3f}')
        loss, acc = model.evaluate(val_dataset)
        print(f'  - Cross-val data -> loss = {loss:.3f} - acc = {acc:.3f}')
        loss, acc = model.evaluate(test_dataset)
        print(f'  - Testing data   -> loss = {loss:.3f} - acc = {acc:.3f}')

        # save model state
        print(f"Saving model state to {MODEL_SAVE_PATH}")
        model.save(MODEL_SAVE_PATH)
        del model

    if DO_PREDICTIONS:
        # load model from saved state & evaluate performance
        model = load_model(MODEL_SAVE_PATH)
        print('Evaluating model performance...')
        loss, acc = model.evaluate(train_dataset)
        print(f'  - Training data  -> loss = {loss:.3f} - acc = {acc:.3f}')
        loss, acc = model.evaluate(val_dataset)
        print(f'  - Cross-val data -> loss = {loss:.3f} - acc = {acc:.3f}')
        loss, acc = model.evaluate(test_dataset)
        print(f'  - Testing data   -> loss = {loss:.3f} - acc = {acc:.3f}')


if __name__ == '__main__':
    main()


"""
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - "
      f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}")

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

hist = model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=32)
kru.show_plots(hist.history, metric='accuracy')

# evaluate performance
loss, acc = model.evaluate(X_train, y_train)
print(f"Training data -> loss: {loss:.3f} - acc: {acc:.3f}")
loss, acc = model.evaluate(X_test, y_test)
print(f"Testing data  -> loss: {loss:.3f} - acc: {acc:.3f}")

# save model
kru.save_model(model, 'kr_fashion2')
del model

"""