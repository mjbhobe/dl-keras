""" keras1.py - introducing Keras3 with Pytorch backend """
# keras1.py - introducing Keras on Pytorch
import os

# set the Keras backend to Pytorch
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers
from keras import ops

# this looks very much like Keras on Tensorflow code, but we are using Pytorch backend
print(f"Using keras {keras.__version__}")


def build_model():
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model


model = build_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)
print(model.summary())

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.20)

test_scores = model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {test_scores[0]:.3f} - test acc: {test_scores[1]:.3f}")
