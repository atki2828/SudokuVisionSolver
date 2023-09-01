from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def gray_borders(
    img: np.ndarray, pool_of_gray: np.ndarray, percent_borders: float
) -> np.ndarray:
    """
    Adds gray borders to an image.

    Args:
        img (np.ndarray): The input image.
        pool_of_gray (np.ndarray): Pool of gray values to choose from.
        percent_borders (float): The percentage of image size to use as borders.

    Returns:
        np.ndarray: The image with gray borders added.

    """
    assert len(img.shape) == 3, "Requires 3-Dimensional Image"

    height, width, _ = img.shape

    # Calculate the border size based on the percentage
    border_size = int(min(height, width) * (percent_borders / 100))

    # Copy the border regions
    horizontal_top = img[:border_size, :].copy()
    horizontal_bottom = img[height - border_size :, :].copy()
    vertical_left = img[:, :border_size].copy()
    vertical_right = img[:, width - border_size :].copy()

    # Assign the gray pixel values to the border regions
    img[:border_size, :] = np.random.choice(
        pool_of_gray.flatten(), size=horizontal_top.size, replace=True
    ).reshape(horizontal_top.shape)
    img[height - border_size :, :] = np.random.choice(
        pool_of_gray.flatten(), size=horizontal_bottom.size, replace=True
    ).reshape(horizontal_bottom.shape)
    img[:, :border_size] = np.random.choice(
        pool_of_gray.flatten(), size=vertical_left.size, replace=True
    ).reshape(vertical_left.shape)
    img[:, width - border_size :] = np.random.choice(
        pool_of_gray.flatten(), size=vertical_left.size, replace=True
    ).reshape(vertical_left.shape)

    return img


def change_digit_color(image: np.ndarray) -> np.ndarray:
    """
    Changes the color of digits in an image to red.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The image with digit color changed to red.

    """
    assert len(image.shape) == 3, "Requires 3-Dimensional Image"

    # Convert the image to grayscale
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    # Gaussian Blur 3X3 kernel
    blurred_image = cv2.GaussianBlur(gray, (3, 3), 0)

    # Inverse Threshold the dilated image to obtain the digit region
    _, binary_image = cv2.threshold(blurred_image, 160, 255, cv2.THRESH_BINARY_INV)

    # Dilate image to fill out white region more
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Create a mask of the binary image
    mask = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)

    # Set the pixels within the mask to red
    image[np.where((mask == [255, 255, 255]).all(axis=2))] = (255, 0, 0)  # RGB format

    return image


def display_im_grid(img_list: List[np.ndarray]):
    """
    Displays a grid of images.

    Args:
        img_list (List[np.ndarray]): List of images to display.

    Returns:
        None

    """
    dim = 3
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(img_list):
        plt.imshow(img_list[i], cmap="gray")
        plt.title(str(i + 1))
        plt.axis("off")


def create_solved_square(blank_square: np.ndarray, digit_square: np.ndarray):
    solved_square = blank_square.copy()
    red_locations = np.where(digit_square[:, :, 0] > 238)
    solved_square[red_locations] = digit_square[red_locations]
    return solved_square


def create_solved_board(
    unsolved_board_image: np.ndarray,
    unsolved_board_array: np.ndarray,
    solved_board_array: np.ndarray,
    square_dict: OrderedDict,
    fill_image_dict: Dict[Tuple[int], np.ndarray],
) -> np.ndarray:
    """
    Creates a solved board image by replacing the unsolved cells with corresponding filled images.

    Args:
        unsolved_board_image (np.ndarray): The original board image.
        unsolved_board_array (np.ndarray): The unsolved board represented as a 2D NumPy array.
        solved_board_array (np.ndarray): The solved board represented as a 2D NumPy array.
        square_dict (OrderedDict): An ordered dictionary containing the position tuples for each square.
        fill_image_dict (Dict[Tuple[int], np.ndarray]): A dictionary mapping digits to corresponding filled images.

    Returns:
        np.ndarray: The solved board image with filled cells.

    """
    board_bool_array = unsolved_board_array == 0
    solved_board = unsolved_board_image.copy()

    for row_index in range(unsolved_board_array.shape[0]):
        for col_index in range(unsolved_board_array.shape[1]):
            if board_bool_array[row_index, col_index]:
                pos_dict, _ = square_dict.popitem(last=False)
                digit = solved_board_array[row_index, col_index]
                digit_square = fill_image_dict.get(digit)
                blank_square = solved_board[
                    pos_dict[0] : pos_dict[1], pos_dict[2] : pos_dict[3]
                ]
                blended_digit_square = create_solved_square(blank_square, digit_square)
                solved_board[
                    pos_dict[0] : pos_dict[1], pos_dict[2] : pos_dict[3]
                ] = blended_digit_square
            else:
                square_dict.popitem(last=False)

    return solved_board
