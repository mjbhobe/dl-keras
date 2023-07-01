# mnist.py
import random
import pathlib
import numpy as np
import tensorflow as tf
from cl_options import TrainingArgsParser

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Using Tensorflow {tf.__version__}")

# hyper-parameters
IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_CLASSES = 28, 28, 1, 10
# EPOCHS, BATCH_SIZE, VERBOSE, VALIDATION_SPLIT = 200, 128, 1, 0.2
NUM_HIDDEN = 128
MODEL_SAVE_BASE_PATH = pathlib.Path(__file__).parent / "model_state"

# MOD_VER, DO_TRAIN, DO_EVAL, DO_PRED = 1, True, True, True

# our model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import kr_helper_funcs as kru

def get_data():
    # download & prepare the dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(
        f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - "
        f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )

    # X_train = X_train.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS)
    # X_test = X_test.reshape(-1, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    print(
        f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - "
        f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )
    return (X_train, y_train), (X_test, y_test)

def get_model(version=1, dropout=None):
    # ------------------------------------------------------------------------
    # Model Performance
    #   Base Model (model_version=1): epochs 200, batch_size 128, optim SGD
    #       train_acc: 92.4 - val_acc: 92.2
    #   1 hidden layer (model_version=2): epochs 25, batch_size 128, optim SGD
    #       train_acc: 94.0 - val_acc: 93.8
    #   2 hidden layers (model_version=3): epochs 25, batch_size 128, optim SGD
    #       train_acc: 95.24 - val_acc: 95.16
    #   2 hidden layers + dropout (model_version=4): epochs 35, batch_size 128, optim SGD
    #       train_acc: 95.93 - val_acc: 95.87
    #   2 hidden layers + dropout + batch norm (model_version=4): epochs 25, batch_size 128, optim SGD
    #       train_acc:  - val_acc:
    # ------------------------------------------------------------------------
    if version == 1:
        model = Sequential(
            [
                Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,),
                        name="input_layer"),
                Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
            ]
        )
    elif version == 2:
        # add a hidden layer
        model = Sequential(
            [
                Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,),
                        name="input_layer"),
                Dense(NUM_HIDDEN, activation="relu", name="hidden_layer"),
                Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
            ]
        )
    elif version == 3:
        # add 2 hidden layers
        model = Sequential(
            [
                Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,),
                        name="input_layer"),
                Dense(NUM_HIDDEN, activation="relu", name="hidden_layer1"),
                Dense(NUM_HIDDEN, activation="relu", name="hidden_layer2"),
                Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
            ]
        )
    elif version == 4:
        # add 2 hidden layers + dropout
        model = Sequential(
            [
                Flatten(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS,),
                        name="input_layer"),
                Dense(NUM_HIDDEN, activation="relu", name="hidden_layer1"),
                Dropout(dropout),
                Dense(NUM_HIDDEN, activation="relu", name="hidden_layer2"),
                Dropout(dropout),
                Dense(NUM_CLASSES, activation="softmax", name="output_layer"),
            ]
        )

    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )
    return model

def main():
    parser = TrainingArgsParser()
    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        help="Model version. 1=base, 2=1 hidden layer, 3=2 hidden layers, \n" +
            "4=2 hidden layers + dropout, 5=2 hidden layers + dropout + batch norm  (default 1)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="Percentage of train set to use for cross-validation (default=0.2)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch (default 1).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default 0.2).",
    )
    args = parser.parse_args()

    MODEL_SAVE_PATH = MODEL_SAVE_BASE_PATH / f"mnist_{args.model_version}.pkl"
    (X_train, y_train), (X_test, y_test) = get_data()

    if args.train:
        model = get_model(args.model_version, args.dropout)
        print(model.summary())
        if args.verbose == 0:
            print(f"Training model for {args.epochs} epochs with batch_size={args.batch_size}")
            print(f"NOTE: no progress will be reported as you chose --verbose={args.verbose}")
        hist = model.fit(
            X_train,
            y_train,
            validation_split=args.val_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        kru.plot_metrics(hist.history, "Model performance")
        kru.save_model(model, str(MODEL_SAVE_PATH))
        del model

    if args.eval:
        # evaluate performance
        model = kru.load_model(str(MODEL_SAVE_PATH))
        print(model.summary())
        loss, acc = model.evaluate(X_train, y_train)
        print(f"Training set -> loss: {loss:.3f} - acc: {acc:.3f}")
        loss, acc = model.evaluate(X_test, y_test)
        print(f"Test set -> loss: {loss:.3f} - acc: {acc:.3f}")
        del model

    if args.pred:
        # make predictions
        model = kru.load_model(str(MODEL_SAVE_PATH))
        print(model.summary())
        preds = np.argmax(model.predict(X_test), axis=1)
        actuals = np.argmax(y_test, axis=1)
        print("Predictions -> Test dataset")
        print(f"  - first 20 data : {actuals[:20]}")
        print(f"  - first 20 preds: {preds[:20]}")
        num_correct = (actuals == preds).sum()
        print(
            f"We got {num_correct} of {len(X_test)} correct predictions (an accuracy = {float(num_correct)/len(X_test) * 100:.2f}%)!"
        )
        del model

if __name__ == "__main__":
    main()
