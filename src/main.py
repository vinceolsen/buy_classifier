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

def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(dataset, repeats=10):
    # load data
    # trainX, trainy, testX, testy = split_data(dataset)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

def run_experiment2(dataset, labels, training_indices, testing_indices, validation_indices, history_length, repeats=10):
    scores = list()

    for r in range(repeats):
        #training
        correct, incorrect = 0, 0
        for day in training_indices:
            chunk = dataset[day-history_length:day,1:6]
            label = labels[day, 1]
            prediction = model.predict(chunk)
            if prediction == label:
                correct+=1
            else:
                incorrect+=1
        score = (correct / (correct + incorrect)) * 100
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)

if __name__ == "__main__":
    # Set up for Windows and Linux
    dataset_folder = Path("raw_data")
    preprocessed_folder = Path("preprocessed_data")

    # You can process up to 5 datasets
    history_length = 506
    PI = ProcessInput(dataset_folder, preprocessed_folder,
                      max_buy_holding_period=10, num_of_securtities=3, target_roi=0.01, history_length=history_length)

    # Run process_datasets() first to save new csv files
    # If csv files already exist, then use read_preprocessed data

    # Process datasets
    PI.process_datasets()

    # Load datasets from preprocessed_data
    PI.read_preprocessed_data()
    # PI.normalize_data()  # let's try training without this first just to get everything working and see what effect normalizing has

    dataset, labels, training_indices, testing_indices, validation_indices = PI.get_data(.80, .10, .10)

    run_experiment2(dataset, labels, training_indices, testing_indices, validation_indices, history_length)

    # Setup model

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
