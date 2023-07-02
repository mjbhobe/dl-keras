# multi_regression.py - multi-variate regression
import sys, random
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"Using Tensorflow {tf.__version__}")

MODEL_SAVE_BASE_PATH = pathlib.Path(__file__).parent / "model_state"

# tensorflow inputs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
import kr_helper_funcs as kru
from cl_options import TrainingArgsParser


def get_data(num_items=100):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]

    # read data from url
    data = pd.read_csv(url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
    data = data.drop(["origin"], axis=1)
    print(f"Number of null records: {data.isna().sum()}")
    data = data.dropna()

    # split into train/test sets - 80%:20%
    train_dataset = data.sample(frac=0.80, random_state=0)
    test_dataset = data.drop(train_dataset.index)
    return train_dataset, test_dataset


def get_model(layer1):
    model = Sequential(
        [
            layer1,
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation=None),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def main():
    parser = TrainingArgsParser()
    # Hey! You can add more args to parser here before parse_args call
    args = parser.parse_args()

    MODEL_SAVE_PATH = MODEL_SAVE_BASE_PATH / f"multivar_regression.pkl"
    train_dataset, test_dataset = get_data()
    column_names = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    if args.show_sample:
        sns.pairplot(train_dataset[column_names], diag_kind="kde")
    plt.show()

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop("mpg")
    test_labels = test_features.pop("mpg")

    data_normalizer = Normalization(axis=1)
    data_normalizer.adapt(np.array(train_features))

    if args.train:
        model = get_model(data_normalizer)
        print(model.summary())
        if args.verbose == 0:
            print(f"Training model for {args.epochs} epochs with batch_size={args.batch_size}")
            print(f"NOTE: no progress will be reported as you chose --verbose={args.verbose}")
        hist = model.fit(
            train_features,
            train_labels,
            validation_split=args.val_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        kru.plot_metrics(hist.history, "Model performance")
        kru.save_model(model, str(MODEL_SAVE_PATH))
        del model

    if args.pred:
        # make predictions
        model = kru.load_model(str(MODEL_SAVE_PATH))
        print(model.summary())
        y_pred = model.predict(test_features).flatten()
        y_true = test_labels.values
        print(f"Actuals: {y_true[:20]}")
        print(f"Predictions: {y_pred[:20]}")

        # plot errors histogram
        errors = y_pred - y_true
        # plt.hist(errors, bins=30)
        sns.histplot(errors, bins=30, kde="True")
        plt.xlabel("Predictions Error [MPG]")
        plt.ylabel("Count")
        plt.title("Distribution of Errors")
        plt.show()

        del model


if __name__ == "__main__":
    main()
