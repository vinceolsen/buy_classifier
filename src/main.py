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
from process_input import ProcessInput


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


if __name__ == "__main__":

    # Set up for Windows and Linux
    dataset_folder = Path("raw_data")
    preprocessed_folder = Path("preprocessed_data")

    # Process BTC historical data
    infile = dataset_folder / 'BTC-Historical-Price.csv'
    outfile = preprocessed_folder / 'BTC-Historical-Price.csv'
    PI = ProcessInput(infile, outfile)
    PI.read_btc_historical_csv()
    PI.write_btc_historical_csv()

    # Read in datasets file names
    src_in = dataset_folder / "datasets.txt"
    with open(src_in, 'r') as file:
        data_string = file.read().replace('\n', '')
    data_string = data_string.split(",")
    src_list = [dataset_folder / x for x in data_string]

    # Setup data
    # TODO
    # Temporarily only using the first 4 datasets
    dataset = process_datasets(src_list[0:4], outfile)

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
