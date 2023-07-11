from typing import List, Union
from pathlib import Path
import numpy as np
import cv2


def get_all_img_files(dir_path: str) -> List[Path]:
    """
    Returns a list of all image files in the directory specified by `dir_path`.

    Parameters
    ----------
    dir_path : str
        The path of the directory to search for image files.

    Returns
    -------
    List[Path]
        A list of `Path` objects for each image file in `dir_path`.
    """
    return [x for x in Path(dir_path).glob("*") if x.is_file()]


def read_imgs_from_files(files: Union[List[Path], str]) -> List[np.ndarray]:
    """
    Reads and returns the images from the specified files.

    Parameters
    ----------
    files : Union[List[Path], str]
        A list of `Path` objects or a single file path as a string.

    Returns
    -------
    List[np.ndarray]
        A list of `numpy` arrays representing the images from the specified files.
    """
    if isinstance(files, str):
        files = [files]
    return [cv2.imread(str(file)) for file in files]
