"""
kr_helper_funcs.py - generic helper functions that can be used across Tensorflow & Keras DL code

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D

Usage:
    > Copy this file to any folder in your sys.path or leave in project folder
    > In the imports section, import this module as 
        import ke_helper_funcs as kru
"""

# imports & tweaks
import warnings

warnings.filterwarnings("ignore")

import sys
import os

# reduce warnings overload from Tensorflow (errors & fatals only!)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import tensorflow as tf
from tensorflow.keras import backend as K

USING_TF2 = tf.__version__.startswith("2")

# -----------------------------------------------------------------------------------------------
# Generic helper functions
# -----------------------------------------------------------------------------------------------
KRU_FAV_SEED = 43

__version__ = "1.5.0"
__author__ = "Manish Bhobe"


def seed_all(seed=None):
    # to ensure that you get consistent results across runs & machines
    """seed all random number generators to get consistent results
    across multiple runs ON SAME MACHINE - you may get different results
    on a different machine (architecture) & that is to be expected

    @params:
         - seed (optional): seed value that you choose to see everything. Can be None
           (default value). If None, the code chooses a random uint between np.uint32.min
           & np.unit32.max
     @returns:
         - if parameter seed=None, then function returns the randomly chosen seed, else it
           returns value of the parameter passed to the function
    """
    if seed is None:
        # pick a random uint32 seed
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if USING_TF2:
        tf.random.set_seed(seed)
    else:
        tf.compat.v1.set_random_seed(seed)
    # log only error from Tensorflow
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    return seed


def setupSciLabModules():
    """
    setup libraries such as Numpy, Pandas, seaborn etc.
    """
    try:

        def float_formatter(x):
            return "%.4f" % x

        np.set_printoptions(formatter={"float_kind": float_formatter})
        np.set_printoptions(
            threshold=np.inf, suppress=True, precision=4, linewidth=1024
        )
    except NameError:
        # Numpy was not imported, so skip Numpy tweaks
        print("Skipping Numpy tweaks", flush=True)
        pass

    try:
        # @see: https://towardsdatascience.com/8-commonly-used-pandas-display-options-you-should-know-a832365efa95
        pd.set_option(
            "display.max_rows", 100
        )  # display upto these many rows before truncating display
        pd.set_option(
            "display.max_columns", 50
        )  # display these many columns before truncating
        pd.set_option(
            "display.max_colwidth", 60
        )  # max width of each column before truncating
        # pd.set_option('display.precision', 4)       # how many floating point numbers after . (setting same as Numpy)
        pd.set_option(
            "display.float_format", "{:,.4f}".format
        )  # use comma separator when displaying numbers
        pd.set_option("display.max_info_columns", 200)
        pd.set_option("display.max_info_rows", 10)
    except NameError:
        # pandas was not installed
        print("Skipping pandas tweaks", flush=True)
        pass

    try:
        plt.style.use("seaborn-v0_8")
    except NameError:
        # Matplotlib was not imported, so skip Matplotlib tweaks
        print("Skipping Matplotlib tweaks", flush=True)
        pass

    try:
        # sns.set(context='notebook', style='whitegrid', font_scale=1.2)
        sns.set_theme(context="notebook", style="darkgrid", font_scale=1.1)
        """
        sns.set_style(
            {
                "font.sans-serif": [
                    "SF Pro Rounded",
                    "Verdana",
                    "Calibri",
                    "DejaVu Sans",
                ]
            }
        )
        """
    except NameError:
        # Seaborn was not imported, so skip Matplotlib tweaks
        print("Skipping Seaborn tweaks", flush=True)
        pass


def progbar_msg(curr_tick, max_tick, head_msg, tail_msg, final=False):
    # --------------------------------------------------------------------
    # Keras like progress bar, used when copying files to show progress
    # --------------------------------------------------------------------
    progbar_len = 30
    len_max_tick = len(str(max_tick))

    if not final:
        prog = (curr_tick * progbar_len) // max_tick
        bal = progbar_len - (prog + 1)
        prog_msg = "  %s (%*d/%*d) [%s%s%s] %s%s" % (
            head_msg,
            len_max_tick,
            curr_tick,
            len_max_tick,
            max_tick,
            "=" * prog,
            ">",
            "." * bal,
            tail_msg,
            " " * 35,
        )
        print("\r%s" % prog_msg, end="", flush=True)
    else:
        prog_msg = "  %s (%*d/%*d) [%s] %s%s\n" % (
            head_msg,
            len_max_tick,
            max_tick,
            len_max_tick,
            max_tick,
            "=" * progbar_len,
            tail_msg,
            " " * 35,
        )
        print("\r%s" % prog_msg, end="", flush=True)


def show_plots(history, metric=None, plot_title=None, fig_size=None):
    import seaborn as sns

    """ Useful function to view plot of loss values & 'metric' across the various epochs
        Works with the history object returned by the fit() or fit_generator() call """
    assert type(history) is dict

    # we must have at least loss in the history object
    assert (
        "loss" in history.keys()
    ), f"ERROR: expecting 'loss' as one of the metrics in history object"
    if metric is not None:
        assert isinstance(
            metric, str
        ), "ERROR: expecting a string value for the 'metric' parameter"
        assert metric in history.keys(), f"{metric} is not tracked in training history!"

    loss_metrics = ["loss"]
    if "val_loss" in history.keys():
        loss_metrics.append("val_loss")
    # after above lines, loss_metrics = ['loss', 'val_loss']

    other_metrics = []
    if metric is not None:
        other_metrics.append(metric)
        if f"val_{metric}" in history.keys():
            other_metrics.append(f"val_{metric}")
    # if metric is not None (e.g. if metrics = 'accuracy'), then other_metrics = ['accuracy', 'val_accuracy']

    # display the plots
    col_count = 1 if len(other_metrics) == 0 else 2
    df = pd.DataFrame(history)

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style(
            {"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]}
        )

        f, ax = plt.subplots(
            nrows=1,
            ncols=col_count,
            figsize=((16, 5) if fig_size is None else fig_size),
        )
        axs = ax[0] if col_count == 2 else ax

        # plot the losses
        losses_df = df.loc[:, loss_metrics]
        losses_df.plot(ax=axs)
        # ax[0].set_ylim(0.0, 1.0)
        axs.grid(True)
        losses_title = (
            "Training 'loss' vs Epochs"
            if len(loss_metrics) == 1
            else "Training & Validation 'loss' vs Epochs"
        )
        axs.title.set_text(losses_title)

        # plot the metric, if specified
        if metric is not None:
            metrics_df = df.loc[:, other_metrics]
            metrics_df.plot(ax=ax[1])
            ax[1].set_ylim(0.0, 1.0)
            ax[1].grid(True)
            metrics_title = (
                f"Training '{other_metrics[0]}' vs Epochs"
                if len(other_metrics) == 1
                else f"Training & Validation '{other_metrics[0]}' vs Epochs"
            )
            ax[1].title.set_text(metrics_title)

        if plot_title is not None:
            plt.suptitle(plot_title)

        plt.show()
        plt.close()


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def time_taken_as_str(start_time, end_time):
    secs_elapsed = end_time - start_time

    SECS_PER_MIN = 60
    SECS_PER_HR = 60 * SECS_PER_MIN

    hrs_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_HR)
    mins_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_MIN)

    if hrs_elapsed > 0:
        ret = "%d hrs %d mins %d secs" % (hrs_elapsed, mins_elapsed, secs_elapsed)
    elif mins_elapsed > 0:
        ret = "Time taken: %d mins %d secs" % (mins_elapsed, secs_elapsed)
    elif secs_elapsed > 1:
        ret = "Time taken: %d secs" % (secs_elapsed)
    else:
        ret = "Time taken - less than 1 sec"
    return ret


def save_model(model, file_path):
    """save model structure & weights to path provided"""
    save_dir, save_filename = os.path.split(file_path)

    if (save_dir != "") and (not os.path.exists(save_dir)):
        # create directory from file_path, if it does not exist
        # e.g. if file_path = '/home/user_name/keras/model_states/model.hd5' and the
        # '/home/user_name/keras/model_states' does not exist, it is created
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print(f"Unable to create folder {save_dir} to save model!")
            raise err

    # now save the model to file_path
    model.save(file_path)
    print(f"Keras model saved to {file_path}")


def load_model(file_path, custom_metrics_map=None):
    """load Keras model from path"""
    # from tensorflow.keras import models

    if not os.path.exists(file_path):
        raise IOError(f"Cannot load Keras model {file_path} - invalid path!")

    # load the state/weights etc. from file_path
    # @see: https://github.com/keras-team/keras/issues/3911
    # useful when you have custom metrics
    # model = models.load_model(file_path, custom_objects=custom_metrics_map)
    model = tf.keras.models.load_model(file_path)
    print(f"Keras model loaded from {file_path}")
    return model


def extract_files(arch_path, to_dir="."):
    """extracts all files from a archive file (zip, tar. tar.bz2 file)
    at arch_path to the 'to_dir' directory"""
    import os
    import tarfile
    import zipfile

    if os.path.exists(arch_path):
        supported_extensions = [".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".npz"]
        arch_exts = [arch_path.endswith(ext) for ext in supported_extensions]

        if np.any(arch_exts):
            # if extension is any of our supported extension, we are ok
            opener_triplets = [
                ("zipfile.ZipFile", zipfile.ZipFile, "r"),
                ("tarfile.open (for .tar.gz or .tgz file)", tarfile.open, "r:gz"),
                ("tarfile.open (for .tar.bz2 or .tbz file)", tarfile.open, "r:bz2"),
            ]
            opened_successfully = False
            curr_dir = os.getcwd()
            os.chdir(to_dir)

            try:
                for opener_str, opener, mode in opener_triplets:
                    try:
                        # try various options to open archive file
                        with opener(arch_path, mode) as f:
                            opened_successfully = True
                            print(
                                f"Extracting files from archive using {opener_str} opener...",
                                flush=True,
                            )
                            f.extractall()
                            break
                    except:
                        continue
            finally:
                os.chdir(curr_dir)
                if not opened_successfully:
                    raise ValueError(
                        f"Could not extract '{arch_path}' as no appropriate extractor is found"
                    )
        else:
            raise ValueError(
                f"Unsupported archive file {arch_path} - only one of {supported_extensions} supported"
            )
    else:
        raise ValueError(f"{arch_path} - path does not exist!")


# ----------------------------------------------------------------------------------------
# custom metrics that can be tracked
# ----------------------------------------------------------------------------------------


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())
