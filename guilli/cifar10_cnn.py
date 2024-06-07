""" cifar10_cnn.py - multi-class classification for CIFAR10 images dataset
    using a CNN """

import warnings

warnings.filterwarnings("ignore")

import sys, os

# reduce warnings overload from Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kr_helper_funcs as kru
from cl_options import TrainingArgsParser

SEED = kru.seed_all(43)
kru.setupSciLabModules()

print(f"Using Tensorflow: {tf.__version__}.")

# globals
IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS = 28, 28, 1
FLATTENED_SHAPE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
NUM_CLASSES = 10
MODEL_SAVE_NAME = "keras_cifar10_cnn.keras"
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "model_states" / MODEL_SAVE_NAME
assert os.path.exists(
    str(MODEL_SAVE_PATH.parent)
), f"FATAL model state save path does not exist - {MODEL_SAVE_PATH.parent}!"


def get_data(val_split=0.2):
    # load & pre-process the data
    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(
        f"Loaded data: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
        f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=SEED
    )

    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")
    X_test = X_test.astype("float32")

    # get mean & std to normalize
    eps = 1e-7
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    # normalize
    X_train = (X_train - mean) / (std + eps)
    X_val = (X_val - mean) / (std + eps)
    X_test = (X_test - mean) / (std + eps)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(
        f"After reshape: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
        f"-  X_val.shape: {X_val.shape} - y_val.shape: {y_val.shape} "
        f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


N_HIDDEN = 128


def build_model(input_shape):
    # adding more layers to base model
    # fmt: off
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"]
    )
    # fmt: on
    return model


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel(
        "Pred {} Conf: {:2.0f}% True ({})".format(
            predicted_label, 100 * np.max(predictions_array), true_label
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array)  # ,color"#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# ------------------------ main code here ------------------------------------

# training hyper-parameters from command line
parser = TrainingArgsParser()
args = parser.parse_args()

X_train, y_train, X_val, y_val, X_test, y_test = get_data(args.val_split)
# sys.exit(-1)

if args.train:
    model = build_model(X_train.shape[1:])
    print(model.summary())

    # use TensorBoard, princess Aurora!
    callbacks = [
        # Write TensorBoard logs to './logs' directory
        tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ]

    print("Training model...")
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        callbacks=[callbacks],
    )
    kru.show_plots(hist.history, metric="accuracy")
    kru.save_model(model, MODEL_SAVE_PATH)
    del model

if args.eval:
    model = kru.load_model(MODEL_SAVE_PATH)

    # evaluate performance
    print("Evaluating performance...")
    loss, acc = model.evaluate(X_train, y_train)
    print(f"  Training data  -> loss: {loss:.4f} - acc: {acc:.4f}")
    loss, acc = model.evaluate(X_val, y_val)
    print(f"  Cross-val data -> loss: {loss:.4f} - acc: {acc:.4f}")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"  Testing data   -> loss: {loss:.4f} - acc: {acc:.4f}")
    del model

if args.pred:
    # run predictions
    model = kru.load_model(MODEL_SAVE_PATH)

    print("Running predictions...")
    predictions = np.argmax(model.predict(X_test), axis=1)
    # we did a to_categorical() call for y_test, reverese it
    y_test = np.argmax(y_test, axis=1)
    # pick random 50
    rand_size = 50
    rand_indexes = np.random.randint(0, len(X_test), rand_size)
    test_labels_sample = y_test[rand_indexes]
    predictions_sample = predictions[rand_indexes]
    print(f"Labels ({rand_size} random): {test_labels_sample}")
    print(f"Predictions for same sample: {predictions_sample}")
    print(
        f"Overall {(y_test == predictions).sum()} correct predictions from {len(y_test)} test records.\n\t"
        f"Accuracy: {((y_test == predictions).sum() / len(y_test)) * 100.0 :.3f} %"
    )
    del model

# -----------------------------------------------------------------------
# Deeper (3 layer) model performance
#   epochs=25, batch_size=32, opt=RMSprop
# Train     -> loss: 0.1104  acc: 0.9637
# Cross-val -> loss: 0.5663  acc: 0.8330
# Test      -> loss: 0.5840  acc: 0.8319
# Analysis:
#  - Model is overfitting (large diff in train & cross-val loss & acc)
# -----------------------------------------------------------------------
