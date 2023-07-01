# sentiment analysis - IBDM dataset
import random
import pathlib
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Using Tensorflow {tf.__version__}")

MODEL_SAVE_BASE_PATH = pathlib.Path(__file__).parent / "model_state"

# tensorflow inputs
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Dense, Embedding, Dropout, GlobalMaxPool1D
import kr_helper_funcs as kru
from cl_options import TrainingArgsParser


def get_data(num_words, max_seq_len):
    # download & prepare the dataset
    (X_train, y_train), (X_test, y_test) = \
        tf.keras.datasets.imdb.load_data(num_words=num_words)
    print(
        f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - "
        f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}"
    )
    # Pad sequences to max_seq_len
    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_seq_len)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_seq_len)

    return (X_train, y_train), (X_test, y_test)

def get_model(vocab_size, embed_dim, max_seq_len):
    # ------------------------------------------------------------------------
    # Model Performance
    #   train_acc: 92.4 - val_acc: 92.2
    # (NOTE: we have used accuracy as dataset is balanced!)
    # ------------------------------------------------------------------------
    model = Sequential(
        [
            Embedding(vocab_size, embed_dim, input_length=max_seq_len),
            Dropout(0.3),
            # Takes the max value of either feature vector from each of
            # the vocab_size features.
            GlobalMaxPool1D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model

def main():
    parser = TrainingArgsParser()
    parser.add_argument(
        "--max-len",
        type=int,
        default=200,
        help="Max length to which sentiments should be padded (default 200)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="Dimension of the embedding layer (default 256)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Size of vocabulary (default 10000)",
    )
    args = parser.parse_args()

    MODEL_SAVE_PATH = MODEL_SAVE_BASE_PATH / f"senti_imdb.pkl"
    (X_train, y_train), (X_test, y_test) = get_data(args.vocab_size, args.max_len)

    if args.train:
        model = get_model(args.vocab_size, args.embed_dim, args.max_len)
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
        preds = (model.predict(X_test) > 0.5).astype(int).ravel()
        actuals = y_test
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
