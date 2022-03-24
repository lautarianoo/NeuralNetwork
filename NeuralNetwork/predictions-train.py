import numpy as np
import pandas as pd

weight_height = pd.read_csv('weight-height.csv')
weight_height = weight_height.sample(frac=1)
weight_height= weight_height.drop(labels=[*range(1000, 10000)], axis=0)
weight_height.loc[(weight_height.Gender == "Female"), 'Gender'] = 0
weight_height.loc[(weight_height.Gender == "Male"), 'Gender'] = 1

wh = weight_height.sample(frac=1)
def mse_loss(y_true, y_pred):
    '''Функция потерь(среднеквадратическая ошибка)'''
    return ((y_true - y_pred)**2).mean()

y_true = np.array([ind for ind in wh['Gender']])
y_pred = np.array([0 for i in range(len(wh['Gender']))])
