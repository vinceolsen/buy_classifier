from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from process_input import ProcessInput

# Set up for Windows and Linux
raw_folder = Path("raw_data")
pre_folder = Path("preprocessed_data")
vis_folder = Path("visualizations")

# Load data from preprocessed_data directory
vis_btc = vis_folder / 'BTC-USD.csv'
btc = pd.read_csv(vis_btc)
print(btc)
