"""
Explorations in Data Science
Crypto buy/sell indicators project
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import random


class ProcessInput:
    def __init__(self, infile, outfile, num_of_securtities=3, max_buy_holding_period=10, target_roi=.01,
                 history_length=506):
        """ Initialize the class

        :param infile: filename of the data to be processed
        :param outfile: filename for the output processed data
        :param num_of_securtities: the number of securities to include in our model input data set, this is the depth or number of layers to the input 3d array
        :param history_length: the number of days of history that we will pass into the model, there are 253 trading days in a year
        """
        self.infile = infile
        self.outfile = outfile
        self.headers = None
        self.dataset_n = num_of_securtities
        self.max_investment_holding_period = max_buy_holding_period
        self.src_list = []
        self.dataset = np.array([])
        self.labels = np.array([])
        self.length = None
        self.target_roi = target_roi
        # self.history_length = history_length
        self.model_input_start_index = history_length
        self.model_input_stop_index = -1

    def read_dataset_list(self):
        """
        Read in datasets names from datasets.txt
        """
        src_in = self.infile / "datasets.txt"
        with open(src_in, 'r') as file:
            data_string = file.read().replace('\n', '')
        self.data_string = data_string.split(",")
        self.src_list = [self.infile / x for x in self.data_string]

    def write_to_csv(self, df, index, visual=False):
        """ Write the processed historical data back to file

        :return: None
        """
        if visual == True:
            visual_dir = Path('visualizations')
            dir = visual_dir / self.data_string[index]
            df.to_csv(dir, index=False)

        dir = self.outfile / self.data_string[index]
        df.to_csv(dir, index=False)

    def write_np_array_to_csv(self, data, name):
        """

        :param data:
        :param name:
        :return:
        """
        name = name + '.csv'
        np.savetxt('preprocessed_data/' + name, data, delimiter=",", fmt='%s')
        np.savetxt('visualizations/' + name, data, delimiter=",", fmt='%s')

    def read_preprocessed_data(self, index=None):
        """
        Read in one or all of the datasets from preprocessed_data 
        directory
        :params: Provide specific index using datasets.txt
        """
        self.read_dataset_list()
        path_list = [self.outfile / x for x in self.data_string]

        # # not currently used
        # if index is not None:
        #     if index > self.dataset_n-1:
        #         raise ValueError("Index value is out of range")
        #
        #     df = pd.read_csv(path_list[index])
        #     self.length = df.shape[0]
        #
        #     # # Generate labels
        #     # self.labels = self.generate_labels(df, labels)
        #     # self.dataset = df.to_numpy()
        #     return

        for name in path_list[:self.dataset_n]:
            df = pd.read_csv(name)

            # Store shape of dataset
            if self.length is None:
                self.length = df.shape[0]

                # Stack data along the 3rd axis
                self.dataset = df.to_numpy()

            # Stack data along the 3rd axis
            data = df.to_numpy()
            print(name, data.shape)

            self.dataset = np.dstack((self.dataset, data))

        # Generate labels
        self.generate_labels()

    def align_initial_dataframes(self):
        """
        Aligns axes of first two datasets for stacking 
        :return: bitcoin price df and df of next dataset on list
        """
        # Read BTC-Historical-Price
        btc_df = pd.read_csv(self.src_list[0])

        # Read 1st US Exchange dataset
        ds_df = pd.read_csv(self.src_list[1])

        # Drop null values from the price
        btc_df = btc_df.dropna()

        # Drop holidays and weekend from BTC price data
        btc_df = self.drop_dates(btc_df, ds_df)

        # Drop dates from financial data where BTC was null
        ds_df = self.drop_dates(ds_df, btc_df)

        # Save csv with dates
        self.write_to_csv(btc_df, 0, visual=True)

        # Drop dates column
        # btc_df = btc_df.drop(['Date'], axis=1)  # let's keep the dates here for troubleshooting and drop them before feeding them into the model
        self.write_to_csv(btc_df, 0)
        # self.length = ds_df.shape[0]

        return ds_df

    def drop_dates(self, df1, df2_ref):
        """
        Drop dates for crypto historical price data
        :return: dataframe
        """
        df1 = df1.set_index(['Date'])
        temp = df2_ref.set_index(['Date'])
        df1 = df1[df1.index.isin(temp.index)].reset_index()
        return df1

    def process_datasets(self):
        """
        Process datasets and stack along 3rd access
        :return: None
        """
        self.read_dataset_list()
        ds_df = self.align_initial_dataframes()

        num = 2
        for set in self.src_list[2:self.dataset_n]:
            df = pd.read_csv(set)

            # Prepare data
            df = self.drop_dates(df, ds_df)
            # df = df.drop(['Date'], axis=1)  # lets keep dates for trouble shooting and slice them off before passing the data to the model
            self.write_to_csv(df, num)
            num += 1

        # ds_df = ds_df.drop(['Date'], axis=1)  # lets keep dates for trouble shooting and slice them off before passing the data to the model
        self.write_to_csv(ds_df, 1)

    def generate_labels(self):
        """
        Generates labels for a dataset
        :params: dataframe, size of series, existing labels (default is none)
        :return: labels
        """

        self.model_input_stop_index = self.length - self.max_investment_holding_period  # -1 for the column headers?

        bitcoin_buy_labels = pd.read_csv('preprocessed_data/BTC-USD.csv')
        # print(bitcoin_buy_labels.shape)
        place_holder_value = -1
        bitcoin_buy_labels = bitcoin_buy_labels.assign(labels=place_holder_value)
        # print(bitcoin_buy_labels.shape)
        # print(bitcoin_buy_labels)
        bitcoin_buy_labels = bitcoin_buy_labels.to_numpy()
        # print(bitcoin_buy_labels.shape)
        # print(bitcoin_buy_labels)

        # iterate over the bitcoin prices to determine actual buy signal labels
        for day in range(self.model_input_start_index, self.model_input_stop_index):
            next_open_price = bitcoin_buy_labels[day + 1, 1]
            # print('next open price', next_open_price)
            prices_during_investment_period = bitcoin_buy_labels[day + 1:day + self.max_investment_holding_period + 1,
                                              1:6]
            # print('day', day, bitcoin_buy_labels[day])
            # print('day1', day + 1, bitcoin_buy_labels[day + 1])
            # print('day2', day + 2, bitcoin_buy_labels[day + 2])
            # print('day3', day + 3, bitcoin_buy_labels[day + 3])
            # print('day4', day + 4, bitcoin_buy_labels[day + 4])
            # print('day5', day + 5, bitcoin_buy_labels[day + 5])
            # print('next10All', bitcoin_buy_labels[day + 1:day + 11])
            # print('next 10 days', prices_during_investment_period.shape, prices_during_investment_period)
            max_price_over_investment_period = prices_during_investment_period.max()
            # print(max_price_over_investment_period)
            buy_label = 1 if max_price_over_investment_period >= (1 + self.target_roi) * next_open_price else 0
            # print(buy_label)
            bitcoin_buy_labels[day, 7] = buy_label
            # print('new row', bitcoin_buy_labels[day])

        # print(bitcoin_buy_labels)
        bitcoin_buy_labels = np.delete(bitcoin_buy_labels, [1, 2, 3, 4, 5, 6], 1)
        # print(bitcoin_buy_labels)
        self.write_np_array_to_csv(bitcoin_buy_labels, 'bitcoin_buy_labels')
        self.labels = bitcoin_buy_labels

    def calculate_change(self, price, series):
        """
        Compare a price to the max value of a list of prices
        :return: resulting value or 
        0 if price is the same or less than max value.
        """
        max_val = np.amax(series)

        if price < max_val:
            change = (max_val - price) / price
            return change
        else:
            return 0

    def normalize_data(self):
        """
        Normalizes each dataset individually using MinMaxScaler
        :return: none
        """

        n = self.dataset.shape[2]
        scaler = {}
        norm_data = []

        for i in range(n):
            scaler[i] = MinMaxScaler()

            # Target a single dataset
            dataset = self.dataset[:, :, i:i + 1]

            # Remove 3rd axis
            dataset = np.squeeze(dataset)

            # First dataset
            if i == 0:
                # Scale and round
                norm_data = scaler[i].fit_transform(dataset)
                norm_data = np.round(norm_data, decimals=11)
                continue

            # Scale and round
            x = scaler[i].fit_transform(dataset)
            x = np.round(x, decimals=11)

            # Restack
            norm_data = np.dstack((norm_data, x))

        self.dataset = norm_data

    def get_data(self, training_portion, testing_portion, validation_portion):
        """ Get the data to pass into the ML model

        :param training_portion: float Percentage of the dataset to use for training
        :param testing_portion: float Percentage of the dataset to use for testing
        :param validation_portion: float Percentage of the dataset to use for validation
        :return: dataset, labels, training_indices, testing_indices, validation_indices
        """
        training_indices = np.array([])
        testing_indices = np.array([])
        validation_indices = np.array([])

        total_indices = self.model_input_stop_index - self.model_input_start_index
        training_count = int(total_indices * training_portion)
        testing_count = int(total_indices * testing_portion)
        validation_count = int(total_indices * validation_portion)

        # print('random',random.sample(
        #     range(self.model_input_start_index, self.model_input_start_index + training_count + 1),
        #     training_count))

        training_indices = np.concatenate((training_indices,
                                           random.sample(
                                               range(self.model_input_start_index,
                                                     self.model_input_start_index + training_count + 1),
                                               training_count)))
        testing_indices = np.concatenate((testing_indices,
                                          random.sample(range(self.model_input_start_index + training_count + 1,
                                                              self.model_input_start_index + training_count + testing_count + 1),
                                                        testing_count)))
        validation_indices = np.concatenate((validation_indices,
                                             random.sample(range(
                                                 self.model_input_start_index + training_count + testing_count + 1,
                                                 self.model_input_start_index + training_count + testing_count + validation_count + 1),
                                                           validation_count)))

        print('training indices', training_indices)
        print('testing indices', testing_indices)
        print('validation indices', validation_indices)
        return self.dataset, self.labels, training_indices, testing_indices, validation_indices
