""" mnist.py - """
import warnings

warnings.filterwarnings('ignore')

import sys, os

# reduce warnings overload from Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import kr_helper_funcs as kru

SEED = kru.seed_all()
kru.setupSciLabModules()

print(f"Using Tensorflow: {tf.__version__}.")

# globals
IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS = 28, 28, 1
FLATTENED_SHAPE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
NUM_CLASSES = 10
MODEL_SAVE_NAME = 'keras_mnist.h5'
MODEL_SAVE_PATH = pathlib.Path(__file__).cwd() / "model_states" / MODEL_SAVE_NAME

# load & pre-process the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Loaded data: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
      f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}")

X_train = X_train.reshape(X_train.shape[0], FLATTENED_SHAPE).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], FLATTENED_SHAPE).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
print(f"After reshape: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
      f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}")

# training hyper-parameters
EPOCHS = 10
BATCH_SIZE = 128
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

DO_TRAINING = False
DO_PREDICTIONS = True

if DO_TRAINING:
    def build_model():
        model = Sequential([
            Dense(512, input_shape = (FLATTENED_SHAPE,), activation = 'relu'),
            Dense(NUM_CLASSES, activation = 'softmax')
        ])
        model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
        return model


    model = build_model()
    print(model.summary())

    print("Training model...")
    hist = model.fit(X_train, y_train, validation_split = VALIDATION_SPLIT,
                     epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2)
    kru.show_plots(hist.history, metric = 'accuracy')

    # evaluate performance
    print('Evaluating performance...')
    loss, acc = model.evaluate(X_train, y_train)
    print(f'  Training data -> loss: {loss:.4f} - acc: {acc:.4f}')
    loss, acc = model.evaluate(X_test, y_test)
    print(f'  Testing data  -> loss: {loss:.4f} - acc: {acc:.4f}')

    kru.save_model(model, MODEL_SAVE_PATH)
    del model

if DO_PREDICTIONS:
    # run predictions
    model = kru.load_model(MODEL_SAVE_PATH)

    print("Running predictions...")
    predictions = np.argmax(model.predict(X_test), axis = 1)
    # we did a to_categorical() call for y_test, reverese it
    y_test = np.argmax(y_test, axis = 1)
    # pick random 50
    rand_size = 50
    rand_indexes = np.random.randint(0, len(X_test), rand_size)
    test_labels_sample = y_test[rand_indexes]
    predictions_sample = predictions[rand_indexes]
    print(f"Labels ({rand_size} random): {test_labels_sample}")
    print(f"Predictions for same sample: {predictions_sample}")
    print(f"Overall {(y_test == predictions).sum()} correct predictions from {len(y_test)} test records.\n\t"
          f"Accuracy: {((y_test == predictions).sum() / len(y_test)) * 100.0 :.3f} %")
    del model
