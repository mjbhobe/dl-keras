#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<< enter file name & purpose >>

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for illustration/educational puroposes ONLY and may not be suitable
for production deployment.
Use code at your own risk!! I am not responsible if your machine explodes :D
"""
import sys, os
import numpy as np
import pandas as pd
import pandas_datareader.data as web

print(f"Using Numpy {np.__version__}, Pandas {pd.__version__}")

df = web.get_data_yahoo("aapl", interval="m") # Apple
print(df.head())
print(df.tail())
df.to_csv('./data/aapl_m.csv')
assert os.path.exists('./data/aapl.csv'), "ERROR: failed to save aapl.csv"







