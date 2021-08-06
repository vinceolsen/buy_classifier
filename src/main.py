"""
Explorations in Data Science
Crypto buy/sell indicators project
"""

import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from tensorflow.keras import layers, models
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
from process_input import ProcessInput


# Update the sys.path to search in the python project directory

def create_model(n_timesteps, n_features, n_outputs):
    """

    :param n_timesteps: The number of days to
    :param n_features:
    :param n_outputs:
    :return:
    """
    verbose, epochs, batch_size = 0, 10, 32
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = models.Sequential()
    # model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, 4)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features, 4)))
    # TODO verify kernel_size for this
    # model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=1, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def create_1d_model(n_timesteps, n_features, n_outputs, number_of_security_datasets):
    """
    :param n_timesteps: The number of days to
    :param n_features:
    :param n_outputs:
    :return:
    """
    verbose, epochs, batch_size = 0, 10, 32
    # n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = models.Sequential()
    layer_coefficients = (8, 64, 256)
    # model.add(layers.Conv1D(
    #     filters=n_features * layer_coefficients[0],
    #     kernel_size=n_timesteps // layer_coefficients[0],
    #     strides=1,
    #     padding="valid",
    #     data_format="channels_last",
    #     dilation_rate=1,
    #     groups=1,
    #     activation='relu',
    #     use_bias=True,
    #     kernel_initializer="glorot_uniform",
    #     bias_initializer="zeros",
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     input_shape=(n_timesteps * n_features, number_of_security_datasets)
    # ))
    model.add(layers.SeparableConv1D(
        filters=n_features * layer_coefficients[0],
        kernel_size=n_timesteps // layer_coefficients[0],
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        depth_multiplier=1,  # increasing this will lead to more depth convolutions
        activation='relu',
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None,
        input_shape=(n_timesteps * n_features, number_of_security_datasets)
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.SeparableConv1D(
        filters=n_features * layer_coefficients[1],
        kernel_size=n_timesteps // layer_coefficients[1],
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        depth_multiplier=1,  # increasing this will lead to more depth convolutions
        activation='relu',
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        pointwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        pointwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        pointwise_constraint=None,
        bias_constraint=None
    ))
    model.add(layers.MaxPooling1D(pool_size=4))
    model.add(layers.Conv1D(
        filters=n_features * layer_coefficients[2],
        kernel_size=n_timesteps // layer_coefficients[2],
        strides=1,
        padding="valid",
        data_format="channels_last",
        dilation_rate=1,
        groups=1,
        activation='relu',
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv1D(
    #     filters=n_features * layer_coefficients[3],
    #     kernel_size=n_timesteps // layer_coefficients[3],
    #     strides=1,
    #     padding="valid",
    #     data_format="channels_last",
    #     dilation_rate=1,
    #     groups=1,
    #     activation='relu',
    #     use_bias=True,
    #     kernel_initializer="glorot_uniform",
    #     bias_initializer="zeros",
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None
    # ))
    # model.add(layers.Dropout(0.1))
    # model.add(layers.Conv1D(
    #     filters=n_features * layer_coefficients[4],
    #     kernel_size=n_timesteps // layer_coefficients[4],
    #     strides=1,
    #     padding="valid",
    #     data_format="channels_last",
    #     dilation_rate=1,
    #     groups=1,
    #     activation='relu',
    #     use_bias=True,
    #     kernel_initializer="glorot_uniform",
    #     bias_initializer="zeros",
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None
    # ))

    model.add(layers.MaxPooling1D(pool_size=8))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def summarize_results(scores):
    """

    :param scores:
    :return:
    """
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(model, dataset, labels, training_indices, testing_indices, validation_indices, history_length,
                   epochs=10):
    """

    :param model: The ML Model
    :param dataset: The dataset to use
    :param labels: The labels for the data
    :param training_indices: List of indices for which data points should be trained on
    :param testing_indices: List of indices for which data points should be tested on
    :param validation_indices: List of indices for which data points should be validated on
    :param history_length: The number of days to include in the history chunk
    :param epochs: How many times should the ML algorithm be trained
    :return: None
    """
    # Add shape to the arrays
    n_attributes = dataset.shape[2]
    # TODO change the 5 to a variable. It comes from the 1:6 slicing in creating chunks below
    chunks = np.empty([len(training_indices), history_length, 5, n_attributes])
    chunk_labels = np.empty([len(training_indices), 1])
    test_chunks = np.empty([len(testing_indices), history_length, 5, n_attributes])
    test_chunk_labels = np.empty([len(testing_indices), 1])
    val_chunks = np.empty([len(validation_indices), history_length, 5, n_attributes])
    val_chunk_labels = np.empty([len(validation_indices), 1])

    # Prep Data
    i = 0
    for day in training_indices:
        day = int(day)
        chunk = dataset[day - history_length:day, 1:6]
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype(('float32'))
        chunks[i] = c
        chunk_labels[i] = label
        i += 1

    # Train model
    history = model.fit(chunks, chunk_labels, epochs=epochs, steps_per_epoch=128,
                        shuffle=False)  # , callbacks=my_callbacks)
    print(history)

    j = 0
    for day in testing_indices:
        day = int(day)
        chunk = dataset[day - history_length:day, 1:6]
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype(('float32'))
        test_chunks[j] = c
        test_chunk_labels[j] = label
        j += 1

    k = 0
    for day in validation_indices:
        day = int(day)
        chunk = dataset[day - history_length:day, 1:6]
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype(('float32'))
        val_chunks[k] = c
        val_chunk_labels[k] = label
        k += 1

    eval_loss = model.evaluate(test_chunks, test_chunk_labels)
    print(eval_loss)

    predictions = np.array([])
    for x in val_chunks:
        predictions = np.concatenate(
            [predictions, np.argmax(model.predict(x.reshape(1, history_length, 5, n_attributes)), axis=-1)])

    # TODO
    print(confusion_matrix(val_chunk_labels, predictions, labels='y_true'))

    print("Train: Number of buy signals: ", list(chunk_labels.flatten()).count(1))
    print("Test Number of buy signals: ", list(test_chunk_labels.flatten()).count(1))
    print("Val Number of buy signals: ", list(val_chunk_labels.flatten()).count(1))


def run_1d_experiment(model, dataset, labels, training_indices, testing_indices, validation_indices, history_length,
                      num_of_features,
                      epochs=10):
    """

    :param model: The ML Model
    :param dataset: The dataset to use
    :param labels: The labels for the data
    :param training_indices: List of indices for which data points should be trained on
    :param testing_indices: List of indices for which data points should be tested on
    :param validation_indices: List of indices for which data points should be validated on
    :param history_length: The number of days to include in the history chunk
    :param epochs: How many times should the ML algorithm be trained
    :return: None
    """
    # Add shape to the arrays
    num_of_security_datasets = dataset.shape[2]
    chunks = np.empty([len(training_indices), history_length * num_of_features, num_of_security_datasets])
    chunk_labels = np.empty([len(training_indices), 1])
    test_chunks = np.empty([len(testing_indices), history_length * num_of_features, num_of_security_datasets])
    test_chunk_labels = np.empty([len(testing_indices), 1])
    val_chunks = np.empty([len(validation_indices), history_length * num_of_features, num_of_security_datasets])
    val_chunk_labels = np.empty([len(validation_indices), 1])

    # Prep Data
    i = 0
    print('num of training inputs:', len(training_indices))
    for day in training_indices:
        day = int(day)
        dataset_history = slice(day - history_length + 1, day + 1)
        chunk = dataset[dataset_history, 1]
        for feature in range(2, 1 + num_of_features):
            chunk = np.concatenate([chunk, dataset[dataset_history, feature]])
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype('float32')
        # print('chunk shape:', c.shape)
        # print('chunks shape:', chunks.shape)
        chunks[i] = c
        chunk_labels[i] = label
        i += 1

    j = 0
    print('num of testing inputs', len(testing_indices))
    for day in testing_indices:
        day = int(day)

        dataset_history = slice(day - history_length + 1, day + 1)
        chunk = dataset[dataset_history, 1]
        for feature in range(2, 1 + num_of_features):
            chunk = np.concatenate([chunk, dataset[dataset_history, feature]])
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype('float32')
        # print('chunk shape:', c.shape)
        # print('chunks shape:', chunks.shape)
        test_chunks[j] = c
        test_chunk_labels[j] = label
        j += 1

    k = 0
    print('num of validation inputs', len(validation_indices))
    for day in validation_indices:
        day = int(day)
        dataset_history = slice(day - history_length + 1, day + 1)
        chunk = dataset[dataset_history, 1]
        for feature in range(2, 1 + num_of_features):
            chunk = np.concatenate([chunk, dataset[dataset_history, feature]])
        label = labels[day, 1]
        # convert values to 'float32' the model complains without it TODO Verify this doesn't ruin the data
        c = np.asarray(chunk).astype('float32')
        val_chunks[k] = c
        val_chunk_labels[k] = label
        k += 1

    # Train model
    # chunks = np.concatenate([chunks, val_chunks])
    # chunk_labels = np.concatenate([chunk_labels, val_chunk_labels])
    repeats = 10
    print('training inputs shape:', chunks.shape)
    chunks = chunks.repeat(repeats, 0)
    chunk_labels = chunk_labels.repeat(repeats)
    print('training inputs repeated shape:', chunks.shape)
    training_batch_size = (len(testing_indices) + len(validation_indices)) // 20
    history = model.fit(chunks,
                        chunk_labels,
                        batch_size=training_batch_size,
                        epochs=epochs,
                        verbose='auto',
                        callbacks=None,
                        validation_split=0.,
                        validation_data=(val_chunks, val_chunk_labels),
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=(len(testing_indices) * repeats) // training_batch_size,
                        validation_steps=None,
                        validation_batch_size=None,
                        validation_freq=1,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=True
                        )  # , callbacks=my_callbacks)
    print(history)

    eval_loss = model.evaluate(test_chunks, test_chunk_labels)
    print(eval_loss)

    predictions = np.array([])
    for x in val_chunks:
        predictions = np.concatenate(
            [predictions,
             np.argmax(model.predict(x.reshape(1, history_length * num_of_features, num_of_security_datasets)),
                       axis=-1)])

    # TODO
    print(confusion_matrix(val_chunk_labels, predictions, labels='y_true'))

    print("Train: Number of buy signals with repeats: ", list(chunk_labels.flatten()).count(1))
    print("Test Number of buy signals: ", list(test_chunk_labels.flatten()).count(1))
    print("Val Number of buy signals: ", list(val_chunk_labels.flatten()).count(1))


if __name__ == "__main__":
    # Set up for Windows and Linux
    dataset_folder = Path("raw_data")
    preprocessed_folder = Path("preprocessed_data")

    history_length = 506
    epochs = 40
    num_of_security_datasets = 22
    num_of_features = 6  # Open,High,Low,Close,Adj Close,Volume
    num_of_outputs = 1  # buy signal yes or no

    # You can process up to 5 datasets
    PI = ProcessInput(dataset_folder, preprocessed_folder,
                      max_buy_holding_period=10, num_of_securtities=num_of_security_datasets-1, target_roi=0.01,
                      history_length=history_length)

    # Run process_datasets() first to save new csv files
    # If csv files already exist, then use read_preprocessed data

    # Process datasets
    PI.process_datasets()

    # Load datasets from preprocessed_data
    PI.read_preprocessed_data()
    # PI.normalize_data()  # let's try training without this first just to get everything working and see what effect normalizing has

    dataset, labels, training_indices, testing_indices, validation_indices = PI.get_data(.80, .10, .10)

    # model = create_model(history_length, 5, 1)
    # run_experiment(model, dataset, labels, training_indices, testing_indices, validation_indices, history_length,
    #                epochs=epochs)
    model = create_1d_model(history_length, num_of_features, num_of_outputs, num_of_security_datasets)
    run_1d_experiment(model, dataset, labels, training_indices, testing_indices, validation_indices, history_length,
                      num_of_features, epochs=epochs)
