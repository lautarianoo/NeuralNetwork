import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

filename = ("https://raw.githubusercontent.com/WISEPLAT/SBER-LSTM-Neural-Network-for-Time-Series-Prediction/master/data/SBER_000101_220128.csv")
df = pd.read_csv(filename, sep=",")
df.rename(columns={'<DATE>': "Date", "<TIME>": "Time", "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
df = df.drop("Time", 1)


class DataLoader():

    def __init__(self, filename, split, cols):
        df = pd.read_csv(filename, sep=",")
        df.rename(
            columns={"<DATE>": "Date", "<TIME>": "Time", "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low",
                     "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        self.df = df.drop('Time', 1)

        i_split = int(len(df) * split)
        self.data_train = df.get(cols).values[:i_split]
        self.data_test = df.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len + 1):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def _next_window(self, i, seq_len, normalise):
        window = self.data_train[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y