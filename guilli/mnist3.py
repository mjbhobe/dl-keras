""" mnist2.py: multiclass digits image classification
    Using multi-layer (3 layer) model & dropout regularization """

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
MODEL_SAVE_NAME = "keras_mnist3.h5"
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "model_states" / MODEL_SAVE_NAME
assert os.path.exists(
    MODEL_SAVE_PATH
), f"FATAL model state save path does not exist - {MODEL_SAVE_PATH}!"

# load & pre-process the data
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(
    f"Loaded data: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
    f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
)

X_train = X_train.reshape(X_train.shape[0], FLATTENED_SHAPE).astype("float32") / 255.0
X_test = X_test.reshape(X_test.shape[0], FLATTENED_SHAPE).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
print(
    f"After reshape: X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} "
    f"-  X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
)

N_HIDDEN = 128

# training hyper-parameters from command line
parser = TrainingArgsParser()
args = parser.parse_args()


def build_model(dropout_rate):
    # adding more layers to base model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                N_HIDDEN,
                input_shape=(FLATTENED_SHAPE,),
                activation="relu",
                name="input_layer_1",
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                N_HIDDEN,
                activation="relu",
                name="hidden_layer_1",
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                N_HIDDEN,
                activation="relu",
                name="hidden_layer_2",
            ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                NUM_CLASSES, name="output_layer_1", activation="softmax"
            ),
        ]
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"]
    )
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
    thisplot = plt.bar(range(10), predictions_array,color"#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


if args.train:
    model = build_model(args.do)
    print(model.summary())

    print("Training model...")
    hist = model.fit(
        X_train,
        y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )
    kru.show_plots(hist.history, metric="accuracy")
    kru.save_model(model, MODEL_SAVE_PATH)
    del model

if args.eval:
    model = kru.load_model(MODEL_SAVE_PATH)

    # evaluate performance
    print("Evaluating performance...")
    loss, acc = model.evaluate(X_train, y_train)
    print(f"  Training data -> loss: {loss:.4f} - acc: {acc:.4f}")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"  Testing data  -> loss: {loss:.4f} - acc: {acc:.4f}")
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
#   epochs=50, batch_size=128, 3 layers, opt=SGD, dropout=0.10
# Train     -> loss: 0.0764  acc: 0.9786
# Test      -> loss: 0.1000  acc: 0.9705
# Analysis:
#  - Model is not overfitting (small diff between
#     train & test accuracies)
#  - Model performance improved slightly (more
#     layers, still better performance?)
# -----------------------------------------------------------------------
