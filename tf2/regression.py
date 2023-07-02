# regression.py - simple linear regression with Python
import random
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
from tensorflow.keras.layers import Dense
import kr_helper_funcs as kru
from cl_options import TrainingArgsParser


def get_data(num_items=100):
    # let's sumulate a univariate house price datasets, with area & price
    np.random.seed(SEED)
    area = 2.5 * np.random.randn(num_items) + 25
    price = 25 * area + 5 + np.random.randint(20, 50, size=len(area))
    data = np.array([area, price])
    data = pd.DataFrame(data.T, columns=["area", "price"])
    return data


def get_model():
    model = Sequential(
        [
            Dense(1, input_shape=(1,), activation=None),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="sgd")
    return model


def main():
    parser = TrainingArgsParser()
    # Hey! You can add more args to parser here before parse_args call
    args = parser.parse_args()

    MODEL_SAVE_PATH = MODEL_SAVE_BASE_PATH / f"univar_regression.pkl"
    data = get_data()
    # display a scatter plot
    sns.scatterplot(data=data, x="area", y="price")
    plt.show()

    # normalize the data - min/max scaler
    data = (data - data.min()) / (data.max() - data.min())

    if args.train:
        model = get_model()
        print(model.summary())
        if args.verbose == 0:
            print(f"Training model for {args.epochs} epochs with batch_size={args.batch_size}")
            print(f"NOTE: no progress will be reported as you chose --verbose={args.verbose}")
        hist = model.fit(
            data["area"],
            data["price"],
            validation_split=args.val_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        kru.plot_metrics(hist.history, "Model performance")
        kru.save_model(model, str(MODEL_SAVE_PATH))
        print(f"y = {model.layers[0].weights} * x + {model.layers[0].bias}")

        del model

    if args.pred:
        # make predictions
        model = kru.load_model(str(MODEL_SAVE_PATH))
        print(model.summary())
        y_pred = model.predict(data["area"])
        print(f"y = {model.layers[0].weights} * x + {model.layers[0].bias}")
        # display plot of prediction
        sns.scatterplot(data=data, x="area", y="price", label="data")
        plt.plot(data["area"], y_pred, lw=2, color="firebrick", label="prediction")
        # plt.xlabel("Area")
        # plt.ylabel("Price")
        plt.legend(loc="best")
        plt.show()
        del model


if __name__ == "__main__":
    main()
