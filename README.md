# Machine Learning & Deep Learning Examples with Keras on Tensorflow
This repository contains several examples of ML & DL using Keras on Tensorflow. 

Written by: Manish Bhobe

License: MIT License<br/>
This code is meant for education purposes only & is not intended for commercial/production use!

**Requirements**:
* Python 3.7 or greater
* Tensorflow > 1.15 (most examples have been tested on Tensorflow 2.0)
* Numpy
* Pandas (for loading CSV datasets)
* Matplotlib & Seaborn plotting libraries
* itertools

**NOTE:** I use Tensorflow's implementation of Keras (i.e. `tf.keras`), so no extra Keras installation needed!

Following files are included:
* `kr_helper_funcs.py` : helper functions for Keras (e.g. `show_plots()`, `save_model()` and `load_model()`)
* `kr_breast_cancer.py`: **binary classification** example on the Wisconsin Breast Cancer dataset
* `kr_iris.py`: **multi-class classification** of the Iris dataset
* `kr_mnist.py`: **multi-class classification** of the MNIST digits dataset using a MLP and a CNN
* `kr_wine.py`: **multi-class classification** of the `sklearn` Wine dataset using MLP
* `kr_regression.py`: **regression** temperature conversion example - predict coefficients of C to F conversion model
* `Keras - Fashion MNIST - CNN.ipynb`: Google Colab compatible notebook file, illustrating **multi-class classification** of the `Fashion MNIST` dataset
* `Keras - Fruits 360 (Kaggle) - CNN.ipynb`: Multi-class classification of `Kaggle Fruits 360` dataset. Illustrates how to **connect from Colab to Kaggle & download dataset** to the Colab notebook.
* `Keras - Malaria Detection (Kaggle) - CNN.ipynb`: Multi-class classification of `Malaria cell images`. Illustrates how to **connect from Colab to Kaggle & download dataset** to the Colab notebook.


