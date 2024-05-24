""" simple_regression.py: simple univariate regression
    with sumulated data """

import warnings

warnings.filterwarnings("ignore")

import sys, os

# reduce warnings overload from Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kr_helper_funcs as kru

SEED = kru.seed_all()
# setup Numpy, Pandas, seaborn etc.
kru.setupSciLabModules()

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

area = 2.5 * np.random.randn(100) + 25
# assume price = 25 * area + random_number
price = 25 * area + (5 + np.random.randint(20, 50, size=len(area)))

# calculate weight & bias (see comment for formula)
w = sum(price*(area-np.mean(area))) / sum((area-np.mean(area))**2)
b = np.mean(price) - w * np.mean(area)
print(f"Parameters -> w: {w:.3f} - b: {b:.3f}")

data = np.array([area, price])
df = pd.DataFrame(data.T, columns=["area", "price"])
fig, ax = plt.subplots()
ax.scatter(df.area, df.price, label="Data")
# plot the prediction
df['price_pred'] = w * df.area + b
ax.plot(df.area, df.price_pred, lw=2, color='firebrick', label="Prediction")
ax.set_xlabel("Area")
ax.set_ylabel("Price")
ax.set_title("House Prices: Area vs Price")
plt.legend(loc='best')
plt.show()
