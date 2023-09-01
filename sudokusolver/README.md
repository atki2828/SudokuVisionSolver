


# SudokuSolver
SudokuSolver is a Python package designed
1. Detect the sudoku board from an image containing a sudoku board
2. Use AI to translate board into 2d array
3. Use a CNN to solve the Sudoku
4. Use Computer Vision to place the solution back on the detected board

# Features
* Detection of Sudoku Puzzles using Contour Detection and Traditional CV Techniques
* Translation of an image of a Sudoku board to a computer readable array
* Train Neural Networks to Detect Images and Solve Sudokus
* Using a CNN solving a Sudoku Puzzle

# Requirements
* Python>=3.7
* numpy==1.23.2
* keras==2.9.0
* tensorflow==2.9.1
* opencv-contrib-python==4.6.0.66

# Installation
pip install git+https://github.com/atki2828/VISIONSUDOKUSOLVER/sudokusolve.git


# Usage
```python
from sudokusolver.detectboard.processimage import extract_sudoku_squares , extract_board, create_board_array

unsolved_board_img = extract_board(sample_game)
display_img(unsolved_board_img, title = 'Extracted Board' , fig_size = (12,10))
```

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.