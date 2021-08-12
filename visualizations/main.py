import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from bollinger_bands import BollingerBands
from moving_average import MovingAverage


if __name__ == "__main__":
    # Set up for Windows and Linux
    data_dir = Path("visualizations/data")
    figure_dir = Path("visualizations/figures")
    test = Path("raw_data")

    # Load data from preprocessed_data directory
    btc_path = data_dir / 'BTC-USD.csv'
    spy_path = data_dir / 'SPY.csv'
    vix_path = data_dir / 'VIX_History.csv'
    eth_path = test / 'ETH-USD.csv'

    btc = pd.read_csv(btc_path)
    spy = pd.read_csv(spy_path)
    vix = pd.read_csv(vix_path)
    eth = pd.read_csv(eth_path)

    boll_bands = BollingerBands(figure_dir)
    boll_bands.generate_figures(btc, sample_size=200, name='BTC')

    ma_graph = MovingAverage(figure_dir)
    ma_graph.generate_sma_figures(
        btc, sample_size=200, ma_days1=20, ma_days2=50, name='BTC')
    ma_graph.generate_ema_figures(
        btc, sample_size=50, ema_days1=5, ema_days2=10, name='BTC')
    ma_graph.generate_combined(
        btc, sample_size=50, ma_days=10, ema_days=5, name='BTC')

    # ma_graph.generate_combined(
    #     spy, sample_size=50, ma_days=10, ema_days=5, name='SPY')

    # ma_graph.generate_combined(
    #     vix, sample_size=300, ma_days=50, ema_days=50, name='VIX')
