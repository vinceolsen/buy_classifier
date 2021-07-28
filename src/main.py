"""
Explorations in Data Science
Crypto buy/sell indicators project
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from process_input import ProcessInput

if __name__ == "__main__":

    # Set up for Windows and Linux
    dataset_folder = Path("raw_data")
    preprocessed_folder = Path("preprocessed_data")

    # You can process up to 5 datasets
    PI = ProcessInput(dataset_folder, preprocessed_folder,
                      series_size=10, dataset_size=3, threshold=0.01)

    # Run process_datasets() first to save new csv files
    # If csv files already exist, then use read_preprocessed data

    # Process datasets
    PI.process_datasets()

    # Load datasets from preprocessed_data
    PI.read_preprocessed_data()
    PI.normalize_data()

    # TODO Split dataset in training and testing sets

    # Setup model
    # TODO Figure out what model to start with. May need to be one we make from scratch

    # Transfer
    # TODO Are we going to use an existing model and transfer?
    # TODO If so, what layers should be added for training?
    """
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    pre_model.trainable = False
    model.compile()
    """

    # Train
    # TODO what optimizer should be used?
    """
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.trainable = True
    model.compile(loss='binary_crossentropy', optimizer=opt)
    """

    # TODO what value should training watch? loss?
    """
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss')
    ]
    history = model.fit(trainer, validation_data=validator, epochs=10, shuffle=False, callbacks=my_callbacks)
    print(history)

    eval_loss = model.evaluate(validator)
    print(eval_loss)

    predictions = np.array([])
    for x, y in validator:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])

    print("part ii")
    print(confusion_matrix(labels, predictions))
    """
