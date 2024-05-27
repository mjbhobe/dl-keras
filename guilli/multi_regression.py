""" multi_regression.py: multi-variate regression with auto-mpg dataset """

import warnings

warnings.filterwarnings("ignore")

import sys, os

# reduce warnings overload from Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import kr_helper_funcs as kru
from cl_options import TrainingArgsParser

SEED = kru.seed_all(43)
# setup Numpy, Pandas, seaborn etc.
kru.setupSciLabModules()

MODEL_SAVE_NAME = "kr_auto_mpg.keras"
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "model_states" / MODEL_SAVE_NAME
DATA_FILE_NAME = "auto_mpg.csv"
DATA_SAVE_PATH = pathlib.Path(__file__).parent / "data" / DATA_FILE_NAME

# download the auto data from UCI rep
from ucimlrepo import fetch_ucirepo

parser = TrainingArgsParser()
args = parser.parse_args()


def get_data():
    if not os.path.exists(DATA_SAVE_PATH):
        # fetch auto-mpg dataset
        # @see: https://archive.ics.uci.edu/dataset/9/auto+mpg + Import Python button
        auto_mpg = fetch_ucirepo(id=9)

        # data (as pandas dataframes)
        X = auto_mpg.data.features
        y = auto_mpg.data.targets

        # our complete dataframe
        data = pd.concat([X, y], axis=1)
        data.to_csv(DATA_SAVE_PATH, index=True)
    else:
        data = pd.read_csv(DATA_SAVE_PATH, index_col=0)

    return data


# fmt: off
# drop null rows
data = get_data()
data = data.dropna()

# randomly split into train/test sets
train_dataset = data.sample(frac=0.8, random_state=SEED)
test_dataset = data.drop(train_dataset.index)
print(f"train_dataset: {len(train_dataset)} rows - test_dataset: {len(test_dataset)} rows")
if args.show_sample:
    print("Generating pairplots of train dataset features. Please wait...", flush=True)
    sns.pairplot(train_dataset[train_dataset.columns.drop("mpg")], diag_kind='kde')
    plt.show()

# extract the features & labels
train_features = train_dataset.drop('mpg', axis=1)
train_labels = train_dataset.pop('mpg')
test_features = test_dataset.drop('mpg', axis=1)
test_labels = test_dataset.pop('mpg')

# normalize the features
Normalization = tf.keras.layers.Normalization
data_normalizer = Normalization(axis=1)
data_normalizer.adapt(np.array(train_features))
# fmt:on


def build_model():
    Sequential = tf.keras.models.Sequential
    Dense = tf.keras.layers.Dense

    model = Sequential(
        [
            data_normalizer,
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation=None),
        ]
    )
    print(model.summary())
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


if args.train:
    model = build_model()
    print(model.summary())
    hist = model.fit(
        x=train_features,
        y=train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
        verbose=args.verbose,
    )
    kru.show_plots(hist.history)
    # kru.save_model(model, MODEL_SAVE_PATH)
    model.save(MODEL_SAVE_PATH)
    del model


model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print(model.summary())
y_pred = model.predict(test_features).flatten()
a = plt.axes(aspect="equal")
plt.scatter(test_labels.values, y_pred)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, lw=2, color="firebrick")
plt.show()
plt.close()
