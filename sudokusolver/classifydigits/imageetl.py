import os
from functools import partial
from pathlib import Path
from typing import Tuple , List , Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    decode_predictions,
    preprocess_input,
)



BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
IMG_HEIGHT, IMG_WIDTH = 32, 32

data_dir = Path('.\\image_classification_training_data')

def get_class_names(data_dir: Path) -> np.ndarray:
    """
    Get sorted class names from a directory.

    Args:
        data_dir: A Path object representing the directory.

    Returns:
        A NumPy array of sorted class names.

    """
    return np.array(sorted([item.name for item in data_dir.glob("*")]))


def plot_comparison(
    original: np.ndarray, filtered: np.ndarray, title_0: str, title_1: str
) -> None:
    """
    Plot a comparison between two images.

    Args:
        original: A NumPy array representing the original image.
        filtered: A NumPy array representing the filtered image.
        title_0: The title for the original image.
        title_1: The title for the filtered image.

    Returns:
        None

    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title(title_0)
    # ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_1)


def decode_img(img: bytes, img_height: int, img_width: int) -> tf.Tensor:
    """
    Decode and resize an image.

    Args:
        img: A bytes object representing the compressed image.
        img_height: The desired height of the output image.
        img_width: The desired width of the output image.

    Returns:
        A TensorFlow tensor representing the decoded and resized image.

    """
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path: str) -> Tuple[tf.Tensor, str]:
    """
    Process a file path by reading, decoding, and obtaining the label.

    Args:
        file_path: The path to the file.

    Returns:
        A tuple containing the processed image as a TensorFlow tensor and the label as a string.

    """
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    # img = tf.image.rgb_to_grayscale(img)
    return img, label


def get_label(file_path: str , class_names: Union[List[str], None] = None) -> tf.Tensor:
    """
    Get the label for a file path.

    Args:
        file_path: The path to the file.

    Returns:
        The label as a TensorFlow tensor.

    """
    # Convert the path to a list of path components
    class_names = get_class_names(data_dir= data_dir) if not class_names else class_names
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)

def configure_for_performance(ds: tf.data.Dataset, batch_size: int, buffer_size: int = 1000) -> tf.data.Dataset:
    """
    Configure a TensorFlow Dataset for performance.

    Args:
        ds: The input TensorFlow Dataset.
        batch_size: The batch size for batching the dataset.
        buffer_size: The buffer size for shuffling the dataset (default: 1000).

    Returns:
        The configured TensorFlow Dataset.

    """
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds