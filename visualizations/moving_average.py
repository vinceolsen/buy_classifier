import matplotlib.pyplot as plt
from pathlib import Path
from numpy import average
import pandas as pd
from matplotlib import pyplot as plt


class MovingAverage:
    def __init__(self, out_dir=None):
        self.out_dir = out_dir

    def generate_averages(self, copy, size):
        length = len(copy)
        for row in range(length):
            if row + size < length:
                mean = copy.values[row:row+size].mean()
                if row == 0:
                    value_list = [round(mean, 2)]
                    continue
                value_list.append(round(mean, 2))

            else:
                value_list.append(float('NaN'))
        return value_list

    def generate_e_averages(self, copy, size):
        multiplier = (2/(size+1))
        length = len(copy)
        for row in range(length):
            open_price = copy.values[row:row+1]
            if row == 0:
                value_list = [open_price]
                continue
            ema = (open_price - value_list[row-1]
                   ) * multiplier + value_list[row-1]
            value_list.append(ema)
        return value_list

    def calculate_moving_average(self, df, size1, size2):

        df['Open'] = df['Open'].round(decimals=2)
        copy = df['Open'].iloc[::-1]

        mean_list1 = self.generate_averages(copy, size1)
        mean_list1.reverse()
        df['MA1'] = mean_list1

        mean_list2 = self.generate_averages(copy, size2)
        mean_list2.reverse()
        df['MA2'] = mean_list2

        return df

    def calculate_e_moving_average(self, df, size1, size2):

        copy = df['Open'].round(decimals=2)
        # copy = df['Open'].iloc[::-1]

        mean_list1 = self.generate_e_averages(copy, size1)
        # mean_list1.reverse()
        df['EMA1'] = mean_list1

        mean_list2 = self.generate_e_averages(copy, size2)
        # mean_list2.reverse()
        df['EMA2'] = mean_list2

        return df

    def combine_averages(self, df, size1, size2):

        copy = df['Open'].round(decimals=2)
        copy_reverse = df['Open'].iloc[::-1].round(decimals=2)

        mean_list1 = self.generate_averages(copy_reverse, size1)
        mean_list1.reverse()
        df['SMA'] = mean_list1

        mean_list2 = self.generate_e_averages(copy, size2)
        # mean_list2.reverse()
        df['EMA'] = mean_list2

        return df

    def generate_sma_figures(self, df, sample_size, ma_days1, ma_days2, name):

        df = self.calculate_moving_average(df, ma_days1, ma_days2)
        latest = len(df['MA1'])
        ma1_column = df['MA1']
        ma2_column = df['MA2']
        open_column = df['Open']

        fig_title = name + ' Opening Price with ' + \
            str(ma_days1) + ' and ' + str(ma_days2) + \
            ' Day Simple Moving Average '

        fig_name = name + '_SMA_' + \
            str(ma_days1) + '_' + str(ma_days2) + \
            '_days_for last_' + str(sample_size) + '.png'

        # plt.figure()
        fig, ax = plt.subplots()
        plt.plot(ma1_column[latest - sample_size:latest], color='r')
        plt.plot(ma2_column[latest - sample_size:latest], color='b')
        plt.plot(open_column[latest - sample_size:latest], color='g')
        legend1 = 'SMA ' + str(ma_days1)
        legend2 = 'SMA ' + str(ma_days2)

        plt.legend([legend1, legend2, 'open'])
        plt.ylabel('Price (USD)')
        plt.xlabel('Day')

        ax.yaxis.set_major_formatter('${x:,}')

        ax.yaxis.set_tick_params(which='major', labelcolor='black',
                                 labelleft=True, labelright=False)
        plt.title(fig_title)

        plt.savefig(self.out_dir / fig_name)

    def generate_ema_figures(self, df, sample_size, ema_days1, ema_days2, name):

        df = self.calculate_e_moving_average(df, ema_days1, ema_days2)
        latest = len(df['EMA1'])
        ma1_column = df['EMA1']
        ma2_column = df['EMA2']
        open_column = df['Open']

        fig_title = name + ' Opening Price with ' + \
            str(ema_days1) + ' and ' + str(ema_days2) + \
            ' Day Exponential Moving Average '

        fig_name = name + '_EMA_' + \
            str(ema_days1) + '_' + str(ema_days2) + \
            '_days_for last_' + str(sample_size) + '.png'

        # plt.figure()
        fig, ax = plt.subplots()

        plt.plot(ma1_column[latest - sample_size:latest], color='r')
        plt.plot(ma2_column[latest - sample_size:latest], color='b')
        plt.plot(open_column[latest - sample_size:latest], color='g')
        legend1 = 'EMA ' + str(ema_days1)
        legend2 = 'EMA ' + str(ema_days2)

        plt.legend([legend1, legend2, 'open'])
        plt.ylabel('Price (USD)')
        plt.xlabel('Day')
        ax.yaxis.set_major_formatter('${x:,}')

        ax.yaxis.set_tick_params(which='major', labelcolor='black',
                                 labelleft=True, labelright=False)

        plt.title(fig_title)

        plt.savefig(self.out_dir / fig_name)

    def generate_combined(self, df, sample_size, ma_days, ema_days, name):

        df = self.combine_averages(df, ma_days, ema_days)
        latest = len(df['SMA'])
        ma_column = df['SMA']
        ema_column = df['EMA']
        open_column = df['Open']

        fig_title = name + ' Opening Price with ' + \
            str(ma_days) + ' day SMA and ' + str(ema_days) + \
            ' day EMA'

        fig_name = name + '_SMA_' + \
            str(ma_days) + '_EMA_' + str(ema_days) + \
            '_for last_' + str(sample_size) + 'days.png'

        # plt.figure()
        fig, ax = plt.subplots()

        ax.plot(ma_column[latest - sample_size:latest], color='r')
        ax.plot(ema_column[latest - sample_size:latest], color='b')
        ax.plot(open_column[latest - sample_size:latest], color='g')
        legend1 = 'SMA ' + str(ma_days)
        legend2 = 'EMA ' + str(ema_days)

        plt.legend([legend1, legend2, 'open'])
        plt.ylabel('Price (USD)')
        plt.xlabel('Day')
        ax.yaxis.set_major_formatter('${x:,}')

        ax.yaxis.set_tick_params(which='major', labelcolor='black',
                                 labelleft=True, labelright=False)
        plt.title(fig_title)

        plt.savefig(self.out_dir / fig_name)
