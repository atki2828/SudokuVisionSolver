import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf


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

