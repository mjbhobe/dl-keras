# ols.py - implement OLS in Tensorflow 2.0
import tensorflow as tf
print(f"Using Tensorflow: {tf.__version__}")

# define the data
X = tf.constant([[1, 0], [1, 2]], tf.float32)
Y = tf.constant([[2], [4]], tf.float32)

# OLS = (X'X)^-1 X'Y
beta_0 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))
beta_1 = tf.matmul(beta_0, tf.transpose(X))
beta = tf.matmul(beta_1, Y)

print(beta.numpy())

# solve using Keras (from tf.keras)
print("Solving using tf.keras...")
ols = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), use_bias=False,
        activation='linear'),
])
ols.compile(optimizer='SGD', loss='mse')
ols.fit(X, Y, epochs=1000, verbose=0)
print(ols.weights[0].numpy())
