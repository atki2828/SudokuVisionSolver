from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# BOARD_RESIZE_DIM = 2520
# BOARD_STEPS = np.arange(0,BOARD_RESIZE_DIM, BOARD_RESIZE_DIM/9.0 , dtype = int)


def convert_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale.

    Parameters
    ----------
    img : np.ndarray
        A `numpy` array representing the image to be converted to grayscale.

    Returns
    -------
    np.ndarray
        A `numpy` array representing the grayscale image.
    """
    img = img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def gaussian_blur(img: np.ndarray, kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Applies Gaussian blur filter to an image

    Parameters:
        img (np.ndarray): The input image
        kernel_size (tuple): The size of the kernel. It must be odd and the width and height must be positive and greater than 0.
    Returns:
        np.ndarray: The image after applying the Gaussian blur filter
    """
    return cv2.GaussianBlur(img, kernel_size, 0)


# # Find the edges in the image using Canny edge detection


def canny_edge(img: np.ndarray, thresh_1: int = 50, thresh_2: int = 200) -> np.ndarray:
    """
    Applies Canny edge detection to an image

    Parameters:
        img (np.ndarray): The input image.
        thresh_1 (int): The first threshold for the hysteresis procedure.
        thresh_2 (int): The second threshold for the hysteresis procedure.
    Returns:
        np.ndarray: The image after applying the Canny edge detection.
    """
    return cv2.Canny(img, thresh_1, thresh_2)


# # Find the contours in the edged image


def find_countours(
    img: np.ndarray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
) -> tuple:
    """
    Finds contours in a binary image

    Parameters:
        img (np.ndarray): The input binary image.
        mode (int): Contour retrieval mode. It has 3 options:  cv2.RETR_EXTERNAL,  cv2.RETR_LIST and  cv2.RETR_CCOMP and cv2.RETR_TREE
        method (int): Contour approximation method. It has 4 options: cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_TC89_KCOS
    Returns:
        tuple: A tuple of (contours, hierarchy) where contours is a list of contours, each represented as a list of points and hierarchy is the output vector, containing information about the image topology.
    """
    return cv2.findContours(img, mode, method)


def get_sorted_contours(
    contours: List[np.ndarray], n_contours: int
) -> List[np.ndarray]:
    """
    Sorts contours based on their area and returns the top 'n_counters' contours

    Parameters:
    contours (List[Union[List[List[int]], np.ndarray]]): A list of contours, each represented as a list of points or numpy ndarray
    n_counters (int): Number of contours that are needed to be returned

    Returns:
    List[Union[List[List[int]], np.ndarray]]: A list of the 'n_counters' contours with the largest areas
    """
    return sorted(contours, key=cv2.contourArea, reverse=True)[:n_contours]


# # Initialize a bounding box for the Sudoku board


def find_board(contours: List[np.ndarray], full_img: np.ndarray) -> np.ndarray:
    """
    Finds the board in the given contours and returns the board image

    Parameters:
        contours (List[np.ndarray]): A list of contours
    Returns:
        np.ndarray: The board image
    """
    board_box = None
    # Loop over the contours
    for contour in contours:
        # Approximate the contour with a polygon
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        # Check if the polygon has four sides (a square)
        if len(polygon) == 4:
            # Save the bounding box of the square
            board_box = cv2.boundingRect(polygon)
            break
    # Extract the Sudoku board from the image
    return full_img[
        board_box[1] : board_box[1] + board_box[3],
        board_box[0] : board_box[0] + board_box[2],
    ]


def resize_img(img: np.ndarray, board_resize_dim: int = 2520) -> np.ndarray:
    """
    Resizes an image to the desired dimensions

    Parameters:
        img (np.ndarray): The input image
        board_resize_dim (int): The desired dimension of the image
    Returns:
        np.ndarray: The resized image
    """
    size = (board_resize_dim, board_resize_dim)
    return cv2.resize(img, size)


def extract_sudoku_squares(
    board_image: np.ndarray, board_resize_dim: int
) -> List[np.ndarray]:
    """
    Extracts individual squares from a Sudoku board image.

    Args:
        board_image: A NumPy array representing the Sudoku board image.
        board_resize_dim: The dimension to resize the board image.

    Returns:
        A list of NumPy arrays, each representing an individual square of the Sudoku board.

    """

    square_dim = int(board_resize_dim / 9)
    board_steps = np.arange(0, board_resize_dim, square_dim)
    im_list = []

    for vert_index in range(len(board_steps)):
        for horiz_index in range(len(board_steps)):
            square_im = board_image[
                board_steps[vert_index] : board_steps[vert_index] + square_dim,
                board_steps[horiz_index] : board_steps[horiz_index] + square_dim,
            ]
            im_list.append(square_im)

    return im_list
