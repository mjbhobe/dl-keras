#!/usr/bin/env python
"""
kr-cat-vs-dog-cleanup.py: cleanup cats vs dog image sets
@see: https://github.com/tensorflow/models/issues/2194
"""
import imghdr
import warnings

warnings.filterwarnings("ignore")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress excessive Tensorflow output

import pathlib, shutil

import tensorflow as tf
import PIL

import kr_helper_funcs as kru

SEED = kru.seed_all()
kru.setupSciLabModules()

IMAGES_FOLDER = os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "PetImages")
IMAGE_WIDTH, IMAGE_HEIGHT = 180, 180
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
GPU_AVAILABLE = tf.config.list_physical_devices('GPU')

print(f"Using Tensorflow {tf.__version__}. GPU {'is available! :)' if GPU_AVAILABLE else 'is NOT available :('}")


def cleanImages():
    total_images, num_processed, num_discarded = 0, 0, 0
    cat_images_processed, cat_images_discarded = 0, 0
    dog_images_processed, dog_images_discarded = 0, 0
    bad_images_folder = pathlib.Path(IMAGES_FOLDER).parent.absolute() / "bad_images"

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
            except IOError as err:
                print(f"Error reading {image_path}: {err}")
                # move the image to bad_images path
                bad_image_path = os.path.join(bad_images_folder, image_name)
                print(f"    Moved corrupt image {image_path} -> {bad_image_path}")
                shutil.move(image_path, bad_image_path)
            finally:
                f.close()

            if not is_jfif:
                num_discarded += 1

                if folder_name == "Cat":
                    cat_images_discarded += 1
                else:
                    dog_images_discarded += 1

                print(f"\n      Bad file - discarding {image_path}", flush = True)
                discard_path = os.path.join(bad_images_folder, image_name)
                shutil.move(image_path, discard_path)
                print(f"      Moved discarded {image_name} -> {discard_path}", flush = True)
            else:
                num_processed += 1

                if folder_name == "Cat":
                    cat_images_processed += 1
                else:
                    dog_images_processed += 1
        print("")

    # display final summary
    print(f"\nCleanup completed: {total_images} images read - {num_processed} kept " +
          f"[{cat_images_processed} cat & {dog_images_processed} dog ] - {num_discarded} " +
          f"discarded [{cat_images_discarded} cat & {dog_images_discarded} dog]")


def cleanUp2():
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(IMAGES_FOLDER, folder_name)
        num_images = len(os.listdir(folder_path))
        print(f"Analysing {num_images} {folder_name} images in {folder_path}...")
        num_discarded = 0
        num_gif2jpg = 0
        num_png2jpg = 0

        image_names = tf.io.gfile.listdir(folder_path)
        for image_name in image_names:
            stem, extension = os.path.splitext(image_name)
            if extension.lower() != ".jpg":
                continue

            image_path = os.path.join(folder_path, image_name)
            print(f"\r  - Processing {image_path}...", end = "", flush = True)

            with tf.io.gfile.GFile(image_path, "rb") as fid:
                encoding = fid.read(4)

            # file has extension .jpg, but could be a PNG or GIF image
            if encoding[0] == 0x89 and encoding[1] == 0x50 and encoding[2] == 0x4e and encoding[3] == 0x47:
                # image is actually a .PNG image with a .jpg extension
                # fix: create a copy of image but save with .png extension, then convert PNG to JPG
                image_path_png = os.path.join(folder_path, stem, ".png")
                tf.io.gfile.copy(image_path, image_path_png)
                PIL.Image.open(image_path_png).convert('RGB').save(image_path, "jpeg")
                print(f"{image_path} was actually a PNG file. Converted to JPG")
                num_png2jpg += 1
            elif encoding[0] == 0x47 and encoding[1] == 0x49 and encoding[2] == 0x46:
                # image is actually a .GIF image with a .jpg extension
                # fix: create a copy of image but save with .gif extension, then convert GIF to JPG
                image_path_gif = os.path.join(folder_path, stem, ".gif")
                tf.io.gfile.copy(image_path, image_path_gif)
                PIL.Image.open(image_path_gif).convert('RGB').save(image_path, "jpeg")
                print(f"{image_path} was actually a GIF file. Converted to JPG")
                num_gif2jpg += 1
            elif encoding[0] != 0xff or encoding[1] != 0xd8 or encoding[2] != 0xff:
                # move to discarded path
                discarded_path = pathlib.Path(image_path).parent.parent / "discarded" / image_name
                tf.io.gfile.copy(image_path, discarded_path)
                tf.io.gfile.remove(image_path)
                print(f"ERROR: {image_path} - unrecognized format. Moving to {discarded_path}", flush = True)

        print("Done!")


def cleanUp():
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(IMAGES_FOLDER, folder_name)
        num_images = len(os.listdir(folder_path))
        print(f"Analysing {num_images} {folder_name} images in {folder_path}...")
        num_discarded = 0
        num_gif2jpg = 0
        num_png2jpg = 0
        num_tiff2jpg = 0

        image_names = tf.io.gfile.listdir(folder_path)
        for image_name in image_names:
            stem, extension = os.path.splitext(image_name)
            if extension.lower() != ".jpg":
                continue

            image_path = os.path.join(folder_path, image_name)
            print(f"\r  - Processing {image_path}...", end = "", flush = True)

            image_type = imghdr.what(image_path)

            if image_type == "png":
                # image is actually a .PNG image with a .jpg extension
                # fix: create a copy of image but save with .png extension, then convert PNG to JPG
                image_path_png = os.path.join(folder_path, stem, ".png")
                tf.io.gfile.copy(image_path, image_path_png)
                PIL.Image.open(image_path_png).convert('RGB').save(image_path, "jpeg")
                print(f"\n{image_path} was actually a PNG file. Converted to JPG")
                num_png2jpg += 1
            elif image_type == "gif":
                # image is actually a .GIF image with a .jpg extension
                # fix: create a copy of image but save with .gif extension, then convert GIF to JPG
                image_path_gif = os.path.join(folder_path, stem, ".gif")
                tf.io.gfile.copy(image_path, image_path_gif)
                PIL.Image.open(image_path_gif).convert('RGB').save(image_path, "jpeg")
                print(f"\n{image_path} was actually a GIF file. Converted to JPG")
                num_gif2jpg += 1
            elif image_type == "tiff":
                # image is actually a .TIFF image with a .jpg extension
                # fix: create a copy of image but save with .gif extension, then convert TIFF to JPG
                image_path_tiff = os.path.join(folder_path, stem, ".tif")
                tf.io.gfile.copy(image_path, image_path_tiff)
                PIL.Image.open(image_path_tiff).convert('RGB').save(image_path, "jpeg")
                print(f"\n{image_path} was actually a GIF file. Converted to JPG")
                num_tiff2jpg += 1
            elif image_type not in ("png", "gif", "tiff", "jpeg"):
                # move to discarded path
                discarded_path = pathlib.Path(image_path).parent.parent.parent / "discarded" / image_name
                tf.io.gfile.copy(image_path, discarded_path, overwrite = True)
                tf.io.gfile.remove(image_path)
                num_discarded += 1
                print(f"\nERROR: {image_path} - unrecognized format. Moving to {discarded_path}", flush = True)

        print(f"\n   Done! {num_images} {folder_path} images analyzed")
        num_kept = num_images - (num_png2jpg + num_gif2jpg + num_tiff2jpg + num_discarded)
        print(f"     {num_kept} images kept and {num_discarded} discarded!")
        print(f"     Of {num_kept} designated JPG images {num_png2jpg} were PNG {num_gif2jpg} were GIF"
              f" and {num_tiff2jpg}  were TIFF images!")


if __name__ == "__main__":
    cleanUp()
