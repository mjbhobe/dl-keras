#!/usr/bin/env python
"""
kr-cat-vs-dog.py: binary classification of cat & dog images
"""
import warnings

warnings.filterwarnings("ignore")

import os

# @see: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# set to 3 (disable ALL - INFO, WARNINGS and ERRORS)
# set to 2 (disable ALL - INFO, and WARNINGS)
# set to 1 (disable ALL - INFO)
# set to 0 (log ALL messages - default)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress excessive Tensorflow output

import pathlib, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Rescaling, Conv2D, BatchNormalization, Activation, add, Dropout,
    MaxPooling2D, Dense, SeparableConv2D, GlobalAveragePooling2D, RandomFlip, RandomRotation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import kr_helper_funcs as kru

# plt.style.use("seaborn")
# sns.set(style="darkgrid", context="notebook", font_scale=1.25)
# sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})
#
# SEED = 1321
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

SEED = kru.seed_all()
kru.setupSciLabModules()

IMAGES_FOLDER = os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "PetImages")
IMAGE_WIDTH, IMAGE_HEIGHT = 180, 180
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
EPOCHS, BATCH_SIZE, LR_RATE = 50, 32, 1e-3
GPU_AVAILABLE = tf.config.list_physical_devices('GPU')

print(f"Using Tensorflow {tf.__version__}. GPU {'is available! :)' if GPU_AVAILABLE else 'is NOT available :('}")


def cleanImages():
    total_images, num_processed, num_discarded = 0, 0, 0
    cat_images_processed, cat_images_discarded = 0, 0
    dog_images_processed, dog_images_discarded = 0, 0

    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(IMAGES_FOLDER, folder_name)
        num_images = len(os.listdir(folder_path))
        print(f"Analysing {num_images} {folder_name} images in {folder_path}...", flush = True)

        for i, image_name in enumerate(os.listdir(folder_path)):
            total_images += 1
            image_path = os.path.join(folder_path, image_name)
            print(f"\r   Image {i} of {num_images}: {image_path}", flush = True, end = "")

            try:
                f = open(image_path, "rb")
                # check if first 10 bytes contain text "JFIF"
                is_jfif = tf.compat.as_bytes("JFIF") in f.peek(10)
            finally:
                f.close()

            if not is_jfif:
                num_discarded += 1

                if folder_name == "Cat":
                    cat_images_discarded += 1
                else:
                    dog_images_discarded += 1

                print(f"\n      Bad file - discarding {image_path}", flush = True)
                discard_path = os.path.join(IMAGES_FOLDER, "discarded", image_name)
                shutil.move(image_path, discard_path)
                print(f"      Moved {image_name} -> {discard_path}", flush = True)
            else:
                num_processed += 1

                if folder_name == "Cat":
                    cat_images_processed += 1
                else:
                    dog_images_processed += 1

    # display final summary
    print(f"\nCleanup completed: {total_images} images read - {num_processed} kept " +
          f"[{cat_images_processed} cat & {dog_images_processed} dog ] - {num_discarded} " +
          f"discarded [{cat_images_discarded} cat & {dog_images_discarded} dog]")


def generate_datasets(image_size = IMAGE_SIZE, batch_size = BATCH_SIZE, val_split = 0.2,
                      shuffle_train = True, shuffle_size = 1024, test_split = None):

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        IMAGES_FOLDER,
        validation_split = val_split,
        subset = "training",
        seed = SEED,
        image_size = image_size,
        batch_size = batch_size)

    if shuffle_train:
        train_dataset.shuffle(shuffle_size, seed = SEED)

    if not GPU_AVAILABLE:
        # apply data augmentation to datasets if GPU is not available
        print("GPU NOT available! Applying data augmentation to train dataset generation code")
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.1),
        ])
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training = True), y))

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        IMAGES_FOLDER,
        validation_split = val_split,
        subset = "validation",
        seed = 1337,
        image_size = image_size,
        batch_size = batch_size)

    # prefetch for better IO performance
    train_dataset = train_dataset.prefetch(buffer_size = BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size = BATCH_SIZE)

    test_dataset = None

    if test_split is not None:
        test_size = int(test_split * val_dataset.__len__().numpy())
        test_dataset = val_dataset.take(test_size)
        val_dataset = val_dataset.skip(test_size)  # remaining

    print(f"Train dataset: {train_dataset.__len__().numpy() * batch_size} images - Cross-val dataset: " +
          f"{val_dataset.__len__().numpy() * batch_size} images - Test dataset: " +
          f"{0 if test_dataset is None else test_dataset.__len__().numpy() * batch_size} images")

    return train_dataset, val_dataset, test_dataset


def display_sample(dataset):
    plt.figure(figsize = (8, 6))
    num_rows, num_cols = 4, 8  # BATCH_SIZE // 2, BATCH_SIZE // 2
    label_text = {0: 'Cat', 1: 'Dog'}

    # display the first batch (we have batch_size = 32)
    for images, labels in dataset.take(1):
        for i in range(BATCH_SIZE):
            ax = plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(label_text[int(labels[i])])
            plt.axis("off")

    plt.show()
    plt.close()


def build_model(input_shape, num_classes):
    inputs = Input(shape = input_shape)
    x = inputs

    if GPU_AVAILABLE:
        # apply augmentation layer if GPU is available
        print("GPU is available! Applying data augmentation to model as layers")
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.1),
        ])
        x = data_augmentation(x)

    # Entry block
    x = Rescaling(1.0 / 255)(x)
    x = Conv2D(32, 3, strides = 2, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding = "same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding = "same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides = 2, padding = "same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides = 2, padding = "same")(
            previous_block_activation
        )
        x = add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(1024, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = Dropout(0.5)(x)
    outputs = Dense(units, activation = activation)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer = Adam(LR_RATE),
        loss = "binary_crossentropy",
        metrics = ["acc"]
    )

    return model


DO_TRAINING = True
DO_PREDICTIONS = False
MODEL_SAVE_NAME = 'kr-cats-vs-dogs.hd5'
MODEL_SAVE_PATH = pathlib.Path(__file__).absolute() / 'model_states' / MODEL_SAVE_NAME


def main():
    # clean image files
    # cleanImages()

    # generate datasets from folders
    train_dataset, val_dataset, test_dataset = \
        generate_datasets(test_split = 0.2)

    # display images from training dataset
    # NOTE: since this dataset is shuffled, we'll get random images
    display_sample(train_dataset)

    if DO_TRAINING:
        # build the model
        model = build_model(input_shape = IMAGE_SIZE + (3,), num_classes = 2)
        print(model.summary())

        callbacks = [
            ModelCheckpoint("save_at_{epoch}.hd5"),
        ]
        hist = model.fit(train_dataset, epochs = EPOCHS, callbacks = callbacks, validation_data = val_dataset)
        kru.show_plots(hist.history, metric = 'acc', plot_title = 'Model performance')

        # evaluate performance
        print("Evaluating...")
        loss, acc = model.evaluate(train_dataset, verbose = 1)
        print(f"  Training dataset  -> loss: {loss:.3f} - acc: {acc:.3f}")
        loss, acc = model.evaluate(val_dataset, verbose = 1)
        print(f"  Cross-val dataset -> loss: {loss:.3f} - acc: {acc:.3f}")
        loss, acc = model.evaluate(test_dataset, verbose = 1)
        print(f"  Testing dataset   -> loss: {loss:.3f} - acc: {acc:.3f}")

        kru.save_model(model, MODEL_SAVE_PATH)
        del model

    if DO_PREDICTIONS:
        # load model
        model = kru.load_model(MODEL_SAVE_PATH)
        print(model.summary())

        y_true, y_pred = np.array([]), np.array([])

        for images, labels in test_dataset.take(1):
            predictions = model.predict(images).numpy()
            predictions = np.argmax(predictions, axis = 1)
            np.append(y_pred, predictions)
            np.append(y_true, labels.numpy())

        print(y_true[:20])
        print(y_pred[:20])


if __name__ == "__main__":
    main()
