""" simple_regression.py: simple univariate regression
    with sumulated data """

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

MODEL_SAVE_NAME = "kr_simple_regression.h5"
MODEL_SAVE_PATH = pathlib.Path(__file__).parent / "model_states" / MODEL_SAVE_NAME

"""
For simple regression, the equation is
    Y = w * X + b
For data with N samples, the coefficients of this equation can be calculated as
    w = {sum(i=1,N) [Y_i * (X_i - mean(X))]} / {sum(i=1,N) [(X_i - mean(X)) ** 2]}
    b = mean(Y) - w * mean(X)
"""

# generate synthetic data for house price, where we assume it depends
# only on area of the house
np.random.seed(SEED)
NUM_ROWS = 100

area = 2.5 * np.random.randn(NUM_ROWS) + 25
# assume price = 25 * area + random_number
price = 25 * area + (5 + np.random.randint(20, 50, size=len(area)))

# calculate weight & bias (see comment for formula)
w = sum(price * (area - np.mean(area))) / sum((area - np.mean(area)) ** 2)
b = np.mean(price) - w * np.mean(area)
print(f"Parameters -> w: {w:.3f} - b: {b:.3f}")

data = np.array([area, price])
df = pd.DataFrame(data.T, columns=["area", "price"])
print(df.head())

# -----------------------------------------------
# Keras model to predict weights & bias
# -----------------------------------------------

# normalize the data
df["area2"] = (df.area - df.area.min()) / (df.area.max() - df.area.min())

# model
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense

# fmt: off
def build_model():
    model = Sequential(
        [
            # NOTE: we want a Linear model, so no activation!
            Dense(1, input_shape=(1,), activation=None)
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="sgd")
    print(model.summary())
    return model
# fmt: on

parser = TrainingArgsParser()
args = parser.parse_args()

if args.train:
    model = build_model()
    hist = model.fit(
        x=df.area2,
        y=df.price,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.val_split,
    )
    kru.show_plots(hist.history)
    # using new Keras API
    kru.save_model(model, MODEL_SAVE_PATH)
    # display weights
    weight, bias = model.layers[0].get_weights()
    print(f"Post training: w = {weight} - b = {bias}")
    del model


# load Keras model weights & bias
# model = build_model()
model = kru.load_model(MODEL_SAVE_PATH)
# model = tf.keras.saving.load_model(MODEL_SAVE_PATH)
print(model.summary())
weight, bias = model.layers[0].get_weights()

print(f"Calculated Params: w = {w:.3f}, b = {b:.3f}")
print(f"Trained Params: w = {weight}, b = {bias}")
# sys.exit(-1)

fig, ax = plt.subplots()
ax.scatter(df.area, df.price, label="Data")
# plot the prediction, using calculated weights & bias
df["price_pred"] = w * df.area + b
# and using Keras model's predictions
model = kru.load_model(MODEL_SAVE_PATH)
df["keras_price_pred"] = model.predict(df["area2"])
# plot both on the same plot - red (calculated), green (Keras model)
ax.plot(df.area, df.price_pred, lw=2, color="firebrick", label="Prediction (calc)")
ax.plot(df.area, df.keras_price_pred, lw=2, color="green", label="Prediction (Keras)")
ax.set_xlabel("Area")
ax.set_ylabel("Price")
ax.set_title("House Prices: Area vs Price")
plt.legend(loc="best")
plt.show()

# now let's do some calculations
NUM_TEST_ROWS = 5
test_areas = 2.5 * np.random.randn(NUM_TEST_ROWS) + 25
print(f"Test Areas: {test_areas.ravel()} ")
prices_pred = w * test_areas + b
print(f"Predicted prices : {prices_pred.ravel()} ")
# now for Keras pred, we need to normalize areas
test_areas_norm = (test_areas - df.area.min()) / (df.area.max() - df.area.min())
# test_areas_norm = test_areas_norm[np.newaxis,:]
print(f"Normalized test areas: {test_areas_norm.ravel()}")
prices_pred_keras = model.predict(test_areas_norm)
print(f"Predicted prices (Keras) : {prices_pred_keras.ravel()} ")
