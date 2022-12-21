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
import pathlib

MODEL_SAVE_NAME = 'kr-cats-vs-dogs.hd5'
MODEL_SAVE_PATH = pathlib.Path(__file__).absolute().parents[0] / 'model_states' / MODEL_SAVE_NAME

print(f"Model save path: {MODEL_SAVE_PATH}")
