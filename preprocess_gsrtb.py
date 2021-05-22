"""
preprocess_gsrtb.py: preprocess GSRTB Traffic Signs image set in a format suitable for
training using a Keras based multi-class classifier

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, random
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
from PIL import Image as Image
import shutil
import pathlib

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set(style='whitegrid', font_scale=1.1)

# Keras imports
import tensorflow as tf
gpus_available = tf.config.list_physical_devices('GPU')
print(f"Using Tensorflow version: {tf.__version__}. " +
      f"GPU {'is available :)' if len(gpus_available) > 0 else 'is NOT available :('}")

# My helper functions for training/evaluating etc.
import kr_helper_funcs as kru

seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

rcParams['axes.titlepad'] = 10

# download the GSRTB image set. This will be downlaoded to ~/.keras/datasets/gsrtb folder
images_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
data_dir = tf.keras.utils.get_file('gsrtb.zip', origin=images_url, untar=True)
data_dir = pathlib.Path(data_dir)
print(f"Downloaded data to {data_dir}")
sys.exit(-1)

arch_path = os.path.join(os.getenv("HOME"), '.keras/datasets/gsrtb.tar.gz')
untar_path = os.path.join(os.getenv("HOME"), '.keras/datasets')
print(f"Extracting {arch_path} to {untar_path}...")
kru.extract_files(arch_path, untar_path)

SRC_IMAGES_PATH = os.path.join(untar_path, 'GTSRB/Final_Training')
print(f"Extracted images available in {SRC_IMAGES_PATH}")

