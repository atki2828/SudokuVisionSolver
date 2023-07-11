import numpy as np
from typing import List


def check_1d(arr: np.ndarray) -> bool:
    """
    Check if a 1D array contains all the numbers from 1 to 9.

    Args:
        arr: A 1D NumPy array.

    Returns:
        A boolean indicating whether the array contains all numbers from 1 to 9.

    """
    return np.all(np.sort(arr) == np.arange(1, 10))


def check_grids(game: np.ndarray) -> bool:
    """
    Check if all 3x3 grids in the given Sudoku game are valid.

    Args:
        game: A 9x9 NumPy array representing a Sudoku game.

    Returns:
        A boolean indicating whether all 3x3 grids in the game are valid.

    """
    for row_index_start in np.arange(0, 7, 3):
        for col_index_start in np.arange(0, 7, 3):
            check_grid = check_1d(
                game[
                    row_index_start : row_index_start + 3,
                    col_index_start : col_index_start + 3,
                ].flatten()
            )
            if not check_grid:
                return False
    return True


def check_valid_sudoku(game: np.ndarray) -> bool:
    """
    Check if the given Sudoku game is valid.

    Args:
        game: A 9x9 NumPy array representing a Sudoku game.

    Returns:
        A boolean indicating whether the game is valid.

    """
    row_check = np.all(np.apply_along_axis(check_1d, axis=1, arr=game))
    col_check = np.all(np.apply_along_axis(check_1d, axis=0, arr=game))
    grid_check = check_grids(game)
    return np.all([grid_check, row_check, col_check])


def check_all_data_set(data_set: np.ndarray) -> bool:
    """
    Check if all Sudoku games in the given dataset are valid.

    Args:
        data_set: A 3D NumPy array representing a dataset of Sudoku games.

    Returns:
        A boolean indicating whether all games in the dataset are valid.

    """
    invalid_game_list: List[int] = []
    for i, game in enumerate(data_set.reshape(-1, 9, 9)):
        if not check_valid_sudoku(game):
            invalid_game_list.append(i)
    if invalid_game_list:
        print(f"Invalid Games at Indices {invalid_game_list}")
        return False
    print("No Invalid Games")
    return True
