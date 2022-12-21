""" kr_linear_regression.py - linear regression on random data """
import warnings

warnings.filterwarnings("ignore")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress excessive Tensorflow output

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import kr_helper_funcs as kru

SEED = kru.seed_all()
kru.setupSciLabModules()

# generate random data for regression
NUM_SAMPLES = 500
SLOPE = 25
BIAS = 5


def get_data(slope = SLOPE, bias = BIAS, num_samples = NUM_SAMPLES, delta = 30):
    area = 2.5 * np.random.randn(num_samples) + 25
    # this is our equation price = m * a + b + e
    # where m = 25, b = 5 + random error
    price = slope * area + bias + np.random.randint(-delta, delta, size = len(area))
    return area, price


area, price = get_data()
data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns = ['area', 'price'])
plt.scatter(data['area'], data['price'])
plt.title(f"Sample Data $y = {SLOPE} * x + {BIAS} + \epsilon$")
plt.show()

# calculate the regression coefficients
W = sum(price * (area - np.mean(area))) / sum((area - np.mean(area)) ** 2)
b = np.mean(price) - W * np.mean(area)
print(f"The regression coefficients are -> W: {W:.2f} & b: {b:.2f}", flush = True)

# plot with line
plt.scatter(data['area'], data['price'])
plt.plot(data['area'], W * data['area'] + b, lw = 3, color = 'orange')
plt.title(f"Regression line: $y = {W:.2f} * x + {b:.2f}$")
plt.show()
