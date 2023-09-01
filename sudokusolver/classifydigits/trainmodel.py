from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def create_model(
    input_shape: Tuple[int, int, int], num_classes: int, compile: bool = True
) -> tf.keras.Model:
    """
    Create a pre-trained VGG16 model with additional classification layers on top.

    Args:
        input_shape: A tuple representing the input shape of the model (height, width, channels).
        num_classes: An integer specifying the number of output classes.

    Returns:
        A compiled Keras model.

    """
    # Load pre-trained VGG16 model
    base_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Add classification layers on top
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    if compile:
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
    return model


def train_save_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    save_file_path: str,
    epochs: int,
    class_weights: dict,
) -> tf.keras.callbacks.History:
    """
    Train the specified model with the provided datasets and save the best weights to the specified file path.

    Args:
        model: A compiled Keras model.
        train_ds: A TensorFlow dataset representing the training data.
        val_ds: A TensorFlow dataset representing the validation data.
        save_file_path: A string representing the file path to save the best weights of the model.
        epochs: An integer specifying the number of training epochs.
        class_weights: A dictionary specifying the class weights for imbalanced classes.

    Returns:
        The training history.

    """
    # Define the checkpoint callback
    checkpoint = ModelCheckpoint(
        save_file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
    )

    # Train the model
    with tf.device("/GPU:0"):
        training = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[checkpoint],
            class_weight=class_weights,
        )

    return training




def create_pred_test_series_data(data_set: tf.data.Dataset,
                                 model: tf.keras.Model,
                                 normalizer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization()) -> Tuple[pd.Series, pd.Series, List[np.ndarray]]:
    """
    Create prediction, label, and image lists from the specified dataset using the provided model.

    Args:
        data_set: A TensorFlow dataset.
        model: A trained Keras model for prediction.
        normalizer: A Keras layer for image normalization (default: BatchNormalization).

    Returns:
        A tuple containing the label series, prediction series, and image list.

    """
    predict_list = []
    label_list = []
    im_list = []
    norm_layer = normalizer

    for batch in data_set:
        pred_matrix = model.predict(batch[0], verbose=2)
        im_list += [np.array(im) for im in norm_layer(batch[0])]
        predict_list += [np.argmax(pred) for pred in pred_matrix]
        label_list += [label.numpy() for label in batch[1]]

    return pd.Series(label_list), pd.Series(predict_list), im_list
