"""
kr_wine.py: Multiclass classification of scikit-learn Wine dataset

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
import matplotlib.pyplot as plt
import seaborn as sns

# tweaks for libraries
np.set_printoptions(precision=6, linewidth=1024, suppress=True)
plt.style.use('seaborn')
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.10)

# Tensorflow & Keras imports
import tensorflow as tf
print(f"Using Tensorflow version: {tf.__version__}")
USING_TF2 = tf.__version__.startswith("2")

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

# My helper functions for training/evaluating etc.
import kr_helper_funcs as kru

# to ensure that you get consistent results across runs & machines
# @see: https://discuss.pytorch.org/t/reproducibility-over-different-machines/63047
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
if USING_TF2:
    tf.random.set_seed(seed)
else:
    tf.compat.v1.set_random_seed(seed)

# log only error from Tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_data(val_split=0.20, test_split=0.10):
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    wine = load_wine()
    X, y, f_names = wine.data, wine.target, wine.feature_names

    # split into train/test sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=val_split, random_state=seed)
    
    # scale data
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)

    # split val dataset into eval & test setssets
    X_val, X_test, y_val, y_test = \
        train_test_split(X_val, y_val, test_size=test_split, random_state=seed)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG = 75, 32, 0.001, 0.005

# our ANN
def build_model(inp_size, hidden1, num_classes):
    l2_reg = l2(L2_REG)

    model = Sequential([
        Dense(units=hidden1, activation='relu', kernel_regularizer=l2_reg,
              input_shape=(inp_size,)),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ])
    opt = SGD(lr=LEARNING_RATE, decay=1e-4, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
    
# class WineNet(pytk.PytkModule):
#     def __init__(self, inp_size, hidden1, num_classes):
#         super(WineNet, self).__init__()
#         self.fc1 = pytk.Linear(inp_size, hidden1)
#         self.relu1 = nn.ReLU()
#         self.out = pytk.Linear(hidden1, num_classes)
#         self.dropout = nn.Dropout(0.20)

#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.dropout(x)
#         # NOTE: nn.CrossEntropyLoss() includes a logsoftmax call, which applies a softmax
#         # function to outputs. So, don't apply one yourself!
#         # x = F.softmax(self.out(x), dim=1)  # -- don't do this!
#         x = self.out(x)
#         return x

DO_TRAINING = True
DO_TESTING = False
DO_PREDICTION = False
MODEL_SAVE_NAME = 'kr_wine_ann'

def main():

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    if DO_TRAINING:
        print('Building model...')
        model = build_model(13, 20, 3)
        # # define the loss function & optimizer that model should
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov=True,
        #                             momentum=0.9, dampening=0, weight_decay=L2_REG)
        # model.compile(loss=criterion, optimizer=optimizer, metrics=['acc'])
        print(model.summary())

        # train model
        print('Training model...')
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        kru.show_plots(hist.history, metric='accuracy', plot_title='Training metrics')

        # evaluate model performance on train/eval & test datasets
        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train, verbose=0)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        oss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        # save model state
        kru.save_model(model, MODEL_SAVE_NAME)
        del model

    if DO_PREDICTION:
        print('\nRunning predictions...')
        # load model state from .pt file
        model = kru.load_model(MODEL_SAVE_NAME)
        print(model.summary())

        print('\nEvaluating model performance...')
        loss, acc = model.evaluate(X_train, y_train)
        print('  Training dataset  -> loss: %.4f - acc: %.4f' % (loss, acc))
        loss, acc = model.evaluate(X_val, y_val)
        print('  Cross-val dataset -> loss: %.4f - acc: %.4f' % (loss, acc))
        oss, acc = model.evaluate(X_test, y_test)
        print('  Test dataset      -> loss: %.4f - acc: %.4f' % (loss, acc))

        y_preds = np.argmax(model.predict(X_test), axis=1)
        # display all predictions
        print(f'Sample labels: {y_test}')
        print(f'Sample predictions: {y_preds}')
        print(f'We got {(y_preds == y_test).sum()}/{len(y_test)} correct!!')

if __name__ == "__main__":
    main()

# --------------------------------------------------
# Results: 
#   MLP with epochs=75, batch-size=32, LR=0.001
#       Training  -> acc: 100%
#       Cross-val -> acc: 96.88%
#       Testing   -> acc: 100%
#  Model is overfitting data!
# --------------------------------------------------