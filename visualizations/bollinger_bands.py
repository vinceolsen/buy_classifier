import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import axes


class BollingerBands:

    def __init__(self, out_dir=None):
        self.out_dir = out_dir

    def bollinger_bands(self, df):
        df['Open'] = df['Open'].round(decimals=2)
        copy = df['Open'].iloc[::-1]
        length = len(copy)

        for row in range(length):
            if row + 20 < length:
                mean = copy.values[row:row+20].mean()
                std = copy.values[row:row+20].std()
                if row == 0:
                    upper = [round(mean + std * 2, 2)]
                    lower = [round(mean - std * 2, 2)]
                    continue
                upper.append(round(mean + std * 2, 2))
                lower.append(round(mean - std * 2, 2))

            else:
                upper.append(float('NaN'))
                lower.append(float('NaN'))

        upper.reverse()
        lower.reverse()
        df['Upper'] = upper
        df['Lower'] = lower

        return df

    def generate_figures(self, df, sample_size, name):

        df = self.bollinger_bands(df)
        latest = len(df['Upper'])
        upper = df['Upper']
        lower = df['Lower']
        open = df['Open']

        fig_title = name + ' Opening Price with Bollinger Bands ' + \
            str(sample_size) + ' days'

        fig_name = name + 'bollinger_bands_' + str(sample_size) + 'days.png'

        # plt.figure()
        fig, ax = plt.subplots()

        ax.plot(upper[latest - sample_size:latest],
                color='b', alpha=0.6, ls='--')
        ax.plot(lower[latest - sample_size:latest],
                color='r', alpha=0.6, ls='--')
        ax.plot(open[latest - sample_size:latest], color='g')

        plt.legend(['upper', 'lower', 'open'])
        plt.ylabel('Price (USD)')

        ax.yaxis.set_major_formatter('${x:,}')

        ax.yaxis.set_tick_params(which='major', labelcolor='black',
                                 labelleft=True, labelright=False)

        plt.xlabel('Day')
        plt.title(fig_title)

        plt.savefig(self.out_dir / fig_name)
