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

def process_datasets(src, btc_src):

    # Read BTC-Historical-Price
    btc_df = pd.read_csv(btc_src, sep=',', names=[
                         "Date", "Price", "Open", "High", "Low", "Vol", "Change"])
    print('Initial shape', btc_df.shape)

    # Read 1st dataset and convert to numpy
    ds_df = pd.read_csv(src[0])

    # Filter BTC dataframe by Date
    # btc_df will match the dates of M-F market datasets
    btc_df = btc_df.set_index(['Date'])
    temp = ds_df.set_index(['Date'])
    btc_df = btc_df[btc_df.index.isin(temp.index)].reset_index()
    print('Resulting shape', btc_df.shape)

    # OPTION 1: DROP Columns so dataset features can match
    # btc_df.drop(['Price', 'Change'], axis=1)

    # BTC prices and Dataset features don't match yet so we don't stack
    # btc = btc_df.to_numpy()
    dataset = ds_df.to_numpy()
    # dataset = np.dstack((btc, dataset))

    # Read remaining datasets and stack them along the 3rd axis
    for set in src[1:]:
        df = pd.read_csv(set)
        data = df.to_numpy()
        dataset = np.dstack((dataset, data))
    print('Loaded datasets shape:', dataset.shape)

    return dataset


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
    trainX, trainy, testX, testy = split_data(dataset)
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


if __name__ == "__main__":

    # Set up for Windows and Linux
    dataset_folder = Path("raw_data")
    preprocessed_folder = Path("preprocessed_data")

    PI = ProcessInput(dataset_folder, preprocessed_folder)

    # Run process_datasets() first to save new csv files
    # If csv files already exist, then use read_preprocessed data

    # Process datasets
    PI.process_datasets()
    # Process BTC historical data
    #infile = dataset_folder / 'BTC-Historical-Price.csv'
    #outfile = preprocessed_folder / 'BTC-Historical-Price.csv'

    # Run process_datasets() first to save new csv files


    # Load datasets from preprocessed_data
    PI.read_preprocessed_data()

    # TODO Split dataset in training and testing sets

    # Setup data
    # Temporarily only using the first 4 datasets
    #dataset = process_datasets(src_list[0:4], outfile)
    #trainX, trainY, textX, testY = split_data(dataset)

    #evaluate_model(trainX, trainy, testX, testy)

    # run the experiment
    run_experiment(dataset)

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
