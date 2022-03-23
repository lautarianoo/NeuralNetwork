import numpy as np
from .datasets import wh

def mse_loss(y_true, y_pred):
    '''Функция потерь(среднеквадратическая ошибка)'''
    return ((y_true - y_pred)**2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([])