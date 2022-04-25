import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

filename = ("https://raw.githubusercontent.com/WISEPLAT/SBER-LSTM-Neural-Network-for-Time-Series-Prediction/master/data/SBER_000101_220128.csv")
df = pd.read_csv(filename, sep=",")
df.rename(columns={'<DATE>': "Date", "<TIME>": "Time", "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close", "<VOL>": "Volume"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
df = df.drop("Time", 1)

split = 0.85
i_split = int(len(df) * split)
cols = ["Close", "Volume"]
data_train = df.get(cols).values[:i_split]
data_test = df.get(cols).values[i_split:]

class DataLoader():

    def __init__(self, filename, split, cols):
        self.df = pd.read_csv(filename)
        i_split = int(len(df) * split)
        self.data_train = df.get(cols).values[:i_split]
        self.data_test = df.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)