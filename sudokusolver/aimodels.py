import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .validate import check_valid_sudoku
from tensorflow.keras.models import Model


def build_cnn_model(
    output_activation: str = "softmax",
    loss: str = "sparse_categorical_crossentropy",
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build and compile a custom Sudoku model.

    Args:
        output_activation: Activation function for the output layer (default: 'softmax').
        loss: Loss function to be optimized during training (default: 'sparse_categorical_crossentropy').
        learning_rate: Learning rate for the Adam optimizer (default: 0.001).

    Returns:
        A compiled Keras model.

    """
    inputs = tf.keras.Input(shape=(9, 9, 1))
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(1024, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(9, 1, activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dense(81 * 9)(x)
    x = tf.keras.layers.LayerNormalization(axis=-1)(x)
    x = tf.keras.layers.Reshape((9, 9, 9))(x)
    outputs = tf.keras.layers.Activation(output_activation)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model


def train_cnn_model(
    checkpoint_path: str, model: tf.keras.Model, X: np.ndarray, y: np.ndarray
) -> tf.keras.callbacks.History:
    """
    Train the model with checkpoint and early stopping callbacks.
    Model will be saved in checkpoint_path location

    Args:
        file_path: A string representing the file path to save the best model weights.

    Returns:
        The training history.

    """
    # Define the checkpoint callback
    X = X.reshape(-1, 9, 9, 1)
    y = y.reshape(-1, 9, 9) - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # Define the early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    # Train the model
    with tf.device("/GPU:0"):
        history = model.fit(
            X_train,
            y_train,
            epochs=10,
            verbose=2,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_callback, early_stop],
        )

    return history, model


# TODO Add Limit for checking
def solver(unsolved_board: np.ndarray, model_solve: Model) -> np.ndarray:
    """
    Solve a Sudoku puzzle using a trained model.

    This function fills the unsolved board iteratively using predictions
    from the provided model.

    Parameters:
    - unsolved_board (np.ndarray): A 9x9 numpy array representing the unsolved Sudoku board.
      Empty squares are indicated by 0.
    - model_solve (Model): A trained TensorFlow model to predict the numbers for Sudoku.

    Returns:
    - np.ndarray: A 9x9 numpy array representing the solved Sudoku board.
    """

    board_array = unsolved_board.copy()
    while np.any(board_array == 0):
        # create index of empty squares
        eligible_squares = np.where(board_array == 0)

        # Create Eligible Check List
        eligible_row, eligible_col = eligible_squares
        eligible_check_list = list(zip(eligible_row, eligible_col))

        # Return probabilities for solutions
        sol = model_solve.predict(board_array.reshape(1, 9, 9, 1),verbose = 0)[0]

        # Select max confidence for eligible spaces
        max_conf = sol[eligible_squares].max()

        # Create an array of bools
        bool_arr = sol == max_conf

        row, col, value = np.where(bool_arr == True)
        positions = list(zip(row, col, value))

        for position in positions:
            if (position[0], position[1]) in eligible_check_list:
                board_array[position[0], position[1]] = position[2] + 1

            else:
                print("Fixed Solution Space")

    if check_valid_sudoku(board_array):
        print("Valid Solution")
    else:
        print("Invalid Solution")

    return board_array


def _guess_invalid(board_array):
    for row in board_array:
        bin_count = np.bincount(row)
        if np.any(bin_count[1:] > 1):
            print('Invalid Guess Try Again')
            return True
    for col in board_array.T:
        bin_count = np.bincount(col)
        if np.any(bin_count[1:] > 1):
            print('Invalid Guess Try Again')
            return True
    return False

def solver_test(unsolved_board: np.ndarray, model_solve: Model) -> np.ndarray:
    """
    Solve a Sudoku puzzle using a trained model.

    This function fills the unsolved board iteratively using predictions
    from the provided model.

    Parameters:
    - unsolved_board (np.ndarray): A 9x9 numpy array representing the unsolved Sudoku board.
      Empty squares are indicated by 0.
    - model_solve (Model): A trained TensorFlow model to predict the numbers for Sudoku.

    Returns:
    - np.ndarray: A 9x9 numpy array representing the solved Sudoku board.
    """

    board_array = unsolved_board.copy()
    while np.any(board_array == 0):
        # create index of empty squares
        eligible_squares = np.where(board_array == 0)

        # Create Eligible Check List
        eligible_row, eligible_col = eligible_squares
        eligible_check_list = list(zip(eligible_row, eligible_col))

        # Return probabilities for solutions
        sol = model_solve.predict(board_array.reshape(1, 9, 9, 1))[0]
        conf_list = list(np.sort(sol[eligible_squares].flatten())[::-1])
        print(conf_list)
        guess_switch = True
        while guess_switch:

            # Select max confidence for eligible spaces
            max_conf = conf_list.pop(0)
            print(f"Length of conf_list = {len(conf_list)}")
            print(f"max_conf = {max_conf}")

            # Create an array of bools
            bool_arr = sol == max_conf

            row, col, value = np.where(bool_arr == True)
            positions = list(zip(row, col, value))

            # Set logic to skip if not in viable position i.e. puzzle came with space filled in
            for position in positions:
                if (position[0], position[1]) in eligible_check_list:
                    board_array[position[0], position[1]] = position[2] + 1
                    break

                else:
                    print("Fixed Solution Space")
            # Set bad guess back to 0
            guess_switch = _guess_invalid(board_array)
            if guess_switch:
                print('invalid guess')
                print(board_array)
                board_array[position[0], position[1]] = 0
                print('resetting guess')
                print(board_array)


    if check_valid_sudoku(board_array):
        print("Valid Solution")
    else:
        print("Invalid Solution")

    return board_array