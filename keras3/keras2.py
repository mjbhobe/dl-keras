""" keras2.py - keras with Pytorch backend """

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import torch

print(f"Using Keras {keras.__version__}")


class CustomModel(keras.Model):
    def train_step(self, data):
        """add custom training step"""
        # type of data will depend on how you load
        # data - in this example we are using Numpy arrays
        X, y = data

        # clear prev gradients
        self.zero_grad()

        # forward pass & loss
        y_pred = self(X, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # compute gradients
        loss.backward()

        trainable_weights = [w for w in self.trainable_weights]
        gradients = [w.value.grad for w in trainable_weights]

        # update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # update all metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # return dict of metrics & calculated values
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """add custom evaluation step"""
        X, y = data
        # run prediction
        y_pred = self(X, training=False)
        # calculate loss
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # calculate other metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # return a dict map of metrics & values
        return {m.name: m.result() for m in self.metrics}


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

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.20)

# evaluate performance
print("Evaluating model performance...")
results = model.evaluate(X_train, y_train)
print(f"Evaluating (train) -> loss: {results[0]:.3f} - acc: {results[1]:.3f}")
results = model.evaluate(X_test, y_test)
print(f"Evaluating (test) -> loss: {results[0]:.3f} - acc: {results[1]:.3f}")

# num_rows = 10_000
# model = build_model()
# X = np.random.random((num_rows, 32))
# y = np.random.random((num_rows, 1))
# model.fit(X, y, epochs=5, validation_split=0.2)
