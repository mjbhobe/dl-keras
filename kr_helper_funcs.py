"""
kr_helper_funcs.py - generic helper functions that can be used across Tensorflow & Keras DL code
@author: Manish Bhobe
Use this code at your own risk. I am not responsible if your computer explodes of GPU gets fried :P

Usage: 
    - to import functions from this module, copy the py file to your project folder
    - import code as follows:
        import kr_helper_funcs as kru
"""

# imports & tweaks
import sys, os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import tensorflow as tf
from tensorflow.keras import backend as K

USING_TF2 = (tf.__version__.startswith('2'))
seed = 123

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

if USING_TF2:
    tf.random.set_seed(seed)
else:
    tf.compat.v1.set_random_seed(seed)

# -----------------------------------------------------------------------------------------------
# Generic helper functions 
# -----------------------------------------------------------------------------------------------
def progbar_msg(curr_tick, max_tick, head_msg, tail_msg, final=False):
    # --------------------------------------------------------------------
    # Keras like progress bar, used when copying files to show progress
    # --------------------------------------------------------------------
    progbar_len = 30
    len_max_tick = len(str(max_tick))

    if not final:
        prog = (curr_tick * progbar_len) // max_tick
        bal = progbar_len - (prog + 1)
        prog_msg = '  %s (%*d/%*d) [%s%s%s] %s%s' % (
            head_msg, len_max_tick, curr_tick, len_max_tick, max_tick, '=' * prog, '>', '.' * bal,
            tail_msg, ' ' * 35)
        print('\r%s' % prog_msg, end='', flush=True)
    else:
        prog_msg = '  %s (%*d/%*d) [%s] %s%s\n' % (
            head_msg, len_max_tick, max_tick, len_max_tick, max_tick, '=' * progbar_len, tail_msg,
            ' ' * 35)
        print('\r%s' % prog_msg, end='', flush=True)
        
def show_plots_(history, plot_title=None, fig_size=None):
    
    import seaborn as sns
    
    """ Useful function to view plot of loss values & accuracies across the various epochs
        Works with the history object returned by the train_model(...) call """
    assert type(history) is dict

    # NOTE: the history object should always have loss & acc (for training data), but MAY have
    # val_loss & val_acc for validation data
    loss_vals = history['loss']
    val_loss_vals = history['val_loss'] if 'val_loss' in history.keys() else None
    
    # accuracy is an optional metric chosen by user
    # NOTE: in Tensorflow 2.0, the keys are 'accuracy' and 'val_accuracy'!! Why Google?? Why!!??
    acc_vals = history['acc'] if 'acc' in history.keys() else None
    if acc_vals is None:
        # try 'accuracy' key, as used by the Tensorflow 2.0 backend
        acc_vals = history['accuracy'] if 'accuracy' in history.keys() else None
        
    assert acc_vals is not None, "Something wrong! Cannot read 'acc' or 'accuracy' from history.keys()"
        
    val_acc_vals = history['val_acc'] if 'val_acc' in history.keys() else None
    if val_acc_vals is None:
        # try 'val_accuracy' key, could be using Tensorflow 2.0 backend!
        val_acc_vals = history['val_accuracy'] if 'val_accuracy' in history.keys() else None    
        
    assert val_acc_vals is not None, "Something wrong! Cannot read 'val_acc' ot 'val_acuracy' from history.keys()"
        
    epochs = range(1, len(history['loss']) + 1)
    
    col_count = 1 if ((acc_vals is None) and (val_acc_vals is None)) else 2
    
    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count, figsize=((16, 5) if fig_size is None else fig_size))
    
        # plot losses on ax[0]
        #ax[0].plot(epochs, loss_vals, color='navy', marker='o', linestyle=' ', label='Training Loss')
        ax[0].plot(epochs, loss_vals, label='Training Loss')
        if val_loss_vals is not None:
            #ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')
            ax[0].plot(epochs, val_loss_vals, label='Validation Loss')
            ax[0].set_title('Training & Validation Loss')
            ax[0].legend(loc='best')
        else:
            ax[0].set_title('Training Loss')
    
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].grid(True)
    
        # plot accuracies, if exist
        if col_count == 2:
            #acc_vals = history['acc']
            #val_acc_vals = history['val_acc'] if 'val_acc' in history.keys() else None

            #ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')
            ax[1].plot(epochs, acc_vals, label='Training Accuracy')
            if val_acc_vals is not None:
                #ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')
                ax[1].plot(epochs, val_acc_vals, label='Validation Accuracy')
                ax[1].set_title('Training & Validation Accuracy')
                ax[1].legend(loc='best')
            else:
                ax[1].set_title('Training Accuracy')

            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].grid(True)
    
        if plot_title is not None:
            plt.suptitle(plot_title)
    
        plt.show()
        plt.close()

    # delete locals from heap before exiting (to save some memory!)
    del loss_vals, epochs, acc_vals
    if val_loss_vals is not None:
        del val_loss_vals
    if val_acc_vals is not None:
        del val_acc_vals

def show_plots(history, metric=None, plot_title=None, fig_size=None):
    
    import seaborn as sns
    
    """ Useful function to view plot of loss values & 'metric' across the various epochs
        Works with the history object returned by the fit() call """
    assert type(history) is dict

    # NOTE: the history object should always have loss & acc (for training data), but MAY have
    # val_loss & val_acc for validation data
    loss_vals = history['loss']
    val_loss_vals = history['val_loss'] if 'val_loss' in history.keys() else None

    # we also show metric, if specified by user
    metric_vals, val_metric_vals = None, None

    if metric is not None:
        assert isinstance(metric, str), "expecting a string value for the \'metric\' parameter"
        assert metric in history.keys(), f"{metric} is not tracked in training history!"
        metric_vals = history[metric]
        # check if validation metrics are also tracked in history (this is optional)
        val_metric_name = f"val_{metric}"
        val_metric_vals = history[val_metric_name] if val_metric_name in history.keys() else None
        
    epochs = range(1, len(history['loss']) + 1)
    
    col_count = 1 if ((metric_vals is None) and (val_metric_vals is None)) else 2
    
    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count, figsize=((16, 5) if fig_size is None else fig_size))
    
        # plot losses on ax[0]
        ax[0].plot(epochs, loss_vals, label='Training \'loss\'')
        if val_loss_vals is not None:
            ax[0].plot(epochs, val_loss_vals, label='Validation \'loss\'')
            ax[0].set_title('Training & Validation \'loss\'')
            ax[0].legend(loc='best')
        else:
            ax[0].set_title('Training \'loss\'')
    
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('\'loss\'')
        ax[0].grid(True)
    
        # plot metric, if specified by user
        if col_count == 2:
            ax[1].plot(epochs, metric_vals, label=f'Training \'{metric}\'')
            if val_metric_vals is not None:
                ax[1].plot(epochs, val_metric_vals, label=f'Validation \'{metric}\'')
                ax[1].set_title(f'Training & Validation \'{metric}\'')
                ax[1].legend(loc='best')
            else:
                ax[1].set_title(f'Training \'{metric}\'')

            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel(f'\'{metric}\'')
            ax[1].grid(True)
    
        if plot_title is not None:
            plt.suptitle(plot_title)
    
        plt.show()
        plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def time_taken_as_str(start_time, end_time):
    secs_elapsed = end_time - start_time

    SECS_PER_MIN = 60
    SECS_PER_HR = 60 * SECS_PER_MIN

    hrs_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_HR)
    mins_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_MIN)

    if hrs_elapsed > 0:
        ret = '%d hrs %d mins %d secs' % (hrs_elapsed, mins_elapsed, secs_elapsed)
    elif mins_elapsed > 0:
        ret = 'Time taken: %d mins %d secs' % (mins_elapsed, secs_elapsed)
    elif secs_elapsed > 1:
        ret = 'Time taken: %d secs' % (secs_elapsed)
    else:
        ret = 'Time taken - less than 1 sec'
    return ret

def save_model(model, base_file_name, save_dir=os.path.join('.', 'model_states')):
    """ save everything to one HDF5 file """
    
    # save the model
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'

    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(base_file_name)) == 0):
        # only file name specified e.g. kr_model.h5. We'll use save_dir to save
        if not os.path.exists(save_dir):
            # check if save_dir exists, else create it
            try:
                os.mkdir(save_dir)
            except OSError as err:
                print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
                raise err
        model_save_path = os.path.join(save_dir, base_file_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = base_file_name
        
    #model_save_path = os.path.join(save_dir, base_file_name)
    model.save(model_save_path)
    print('Saved model to file %s' % model_save_path)
    
def load_model(base_file_name, save_dir=os.path.join('.', 'model_states'), 
               custom_metrics_map=None, use_tf_keras_impl=True):
                
    """load model from HDF5 file"""
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'
        
    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(base_file_name)) == 0):
        # only file name specified e.g. kr_model.h5
        model_save_path = os.path.join(save_dir, base_file_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = base_file_name

    if not os.path.exists(model_save_path):
        raise IOError('Cannot find model state file at %s!' % model_save_path)
        
    # load the state/weights etc.
    if use_tf_keras_impl:
        from tensorflow.keras.models import load_model 
    else:
        from keras.models import load_model

    # load the state/weights etc. from .h5 file        
    # @see: https://github.com/keras-team/keras/issues/3911
    # useful when you have custom metrics
    model = load_model(model_save_path, custom_objects=custom_metrics_map)
    print('Loaded Keras model from %s' % model_save_path)
    return model

def save_model_json(model, base_file_name, save_dir=os.path.join('.','model_states')):
    """ save the model structure to JSON & weights to HD5 """    
    # check if save_dir exists, else create it
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
            raise err
            
    # model structure is saved to $(save_dir)/base_file_name.json
    # weights are saved to $(save_dir)/base_file_name.h5
    model_json = model.to_json()
    json_file_path = os.path.join(save_dir, (base_file_name + ".json"))
    h5_file_path = os.path.join(save_dir, (base_file_name + ".h5"))            

    with open(json_file_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5\n",
    model.save_weights(h5_file_path)
    print("Saved model to files %s and %s" % (json_file_path, h5_file_path))

def load_model_json(base_file_name, load_dir=os.path.join('.', 'keras_models'),
                          use_tf_keras_impl=True):
    """ loads model structure & weights from previously saved state """
    # model structure is loaded $(load_dir)/base_file_name.json
    # weights are loaded from $(load_dir)/base_file_name.h5

    if use_tf_keras_impl:
        from tensorflow.keras.models import model_from_json
    else:
        from keras.models import model_from_json

    # load model from save_path
    loaded_model = None
    json_file_path = os.path.join(load_dir, (base_file_name + ".json"))
    h5_file_path = os.path.join(load_dir, (base_file_name + ".h5"))

    if os.path.exists(json_file_path) and os.path.exists(h5_file_path):
        with open(json_file_path, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(h5_file_path)
        print("Loaded model from files %s and %s" % (json_file_path, h5_file_path))
    else:
        msg = "Model file(s) not found in %s! Expecting to find %s and %s in this directory." % (
            load_dir, (base_file_name + ".json"), (base_file_name + ".h5"))
        raise IOError(msg)
    return loaded_model
  
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
  return 2*((prec*rec)/(prec+rec+K.epsilon()))
