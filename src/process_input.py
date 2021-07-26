"""
Explorations in Data Science
Crypto buy/sell indicators project
"""
import calendar
import csv
import datetime
import pandas as pd
import numpy as np
from pathlib import Path


class ProcessInput:

    def __init__(self, infile, outfile):
        """ Initialize the class

        :param infile: filename of the data to be processed
        :param outfile: filename for the output processed data
        """
        self.infile = infile
        self.outfile = outfile
        self.headers = None
        self.btc_data = []
        self.dataset_n = 3
        self.src_list = []
        self.dataset = []
        self.labels = []
        self.length = None

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

    def read_preprocessed_data(self, index=None):
        """
        Read in one or all of the datasets from preprocessed_data 
        directory
        :params: Provide specific index using datasets.txt
        """
        self.read_dataset_list()
        n = 10
        labels = []
        path_list = [self.outfile / x for x in self.data_string]

        if index is None:
            for name in path_list[:self.dataset_n]:
                df = pd.read_csv(name)

                # Store shape of dataset
                if self.length is None:
                    self.length = df.shape[0]

                # Generate labels
                labels = self.generate_labels(df, n, labels)

                # Stack data along the 3rd axis
                data = df.to_numpy()
                self.dataset = np.dstack((self.dataset, data))

        else:
            df = pd.read_csv(path_list[index])
            self.length = df.shape[0]

            # Generate labels
            labels = self.generate_labels(df, n, labels)
            self.dataset = df.to_numpy()

        self.labels = labels

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
        btc_df = btc_df.drop(['Date'], axis=1)
        self.write_to_csv(btc_df, 0)
        self.length = ds_df.shape[0]

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
            df = df.drop(['Date'], axis=1)
            self.write_to_csv(df, num)
            num += 1

        ds_df = ds_df.drop(['Date'], axis=1)
        self.write_to_csv(ds_df, 1)

    def generate_labels(self, df, n, labels=None):
        """
        Generates labels for a dataset
        :params: dataframe, size of series, existing labels (default is none)
        :return: labels
        """

        threshold = .01
        if labels == None:
            labels = []

        prices = df['Open']
        prices = prices.to_numpy()

        row_num = 0
        for row in prices:
            if row_num + n >= self.length:
                series = prices[row_num:self.length+1]

            else:
                series = prices[row_num:n+row_num+1]

            val = self.calculate_change(row, series)
            labels.append(1) if val >= threshold else labels.append(0)
            row_num += 1

        return labels

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
