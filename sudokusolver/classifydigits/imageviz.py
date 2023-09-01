import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
from .imageetl import get_class_names, data_dir


def plot_comparison(
    original: np.ndarray,
    filtered: np.ndarray,
    title_0: str,
    title_1: str,
    fig_size: Tuple[int] = (12, 8),
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
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=fig_size, sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title(title_0)
    # ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(title_1)


def display_im_grid(data_set: tf.data.Dataset, dim: int) -> None:
    """
    Display a grid of images from a TensorFlow Dataset.

    Args:
        data_set: The TensorFlow Dataset containing image and label batches.
        dim: The dimension of the grid (number of rows and columns).

    Returns:
        None

    Raises:
        StopIteration: If `data_set` does not contain enough elements for the specified grid dimension.

    """
    try:
        image_batch, label_batch = next(iter(data_set))
    except StopIteration:
        raise StopIteration(
            "Not enough elements in the dataset for the specified grid dimension."
        )
    class_names = get_class_names(data_dir=data_dir)

    plt.figure(figsize=(10, 10))
    for i in range(dim ** 2):
        ax = plt.subplot(dim, dim, i + 1)
        plt.imshow(image_batch[i].numpy(), cmap="gray")
        print(image_batch[i].numpy().mean())
        print(image_batch[i].numpy().shape)
        label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")


def display_img(img: np.ndarray, title: str = "", fig_size=(8, 6)) -> None:
    plt.figure(figsize=fig_size)
    plt.imshow(img, cmap="gray")
    plt.title(title)
