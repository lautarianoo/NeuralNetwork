import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Layer, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import datetime as dt
import os

class Model():

    def __init__(self):
        self.model = Sequential()
        self.log_dir = "tf_logs"
        self.writer = None

    def load_model(self, filepath):
        self.model = load_model(filepath)

    def build_model(self, configs):

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'],
                           metrics=['accuracy'])
        self.model.summary()

    def train(self, x, y, epochs, batch_size, save_dir, timeframe):

        save_fname = os.path.join(save_dir, '%s-e%s_%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs), timeframe))
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1),
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)