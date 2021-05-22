import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import seaborn as sns
import kr_helper_funcs as kru

plt.style.use('seaborn')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(f"X_train.shape: {X_train.shape} - y_train.shape: {y_train.shape} - "
      f"X_test.shape: {X_test.shape} - y_test.shape: {y_test.shape}")

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

hist = model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=32)
kru.show_plots(hist.history, metric='accuracy')

# evaluate performance
loss, acc = model.evaluate(X_train, y_train)
print(f"Training data -> loss: {loss:.3f} - acc: {acc:.3f}")
loss, acc = model.evaluate(X_test, y_test)
print(f"Testing data  -> loss: {loss:.3f} - acc: {acc:.3f}")

# save model
kru.save_model(model, 'kr_fashion2')
del model

