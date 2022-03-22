import numpy as np

def sigmoid(x):
    #Функция активации f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    '''Составляющая нейросети'''

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforwards(self, inputs):
        #Процесс передачи входов дальше, чтобы получить выход
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

