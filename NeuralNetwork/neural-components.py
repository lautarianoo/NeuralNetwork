import numpy as np

def sigmoid(x):
    #Функция активации f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

class Neuron:
    '''Составляющая нейросети'''

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        #Процесс передачи входов дальше, чтобы получить выход
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class NeuralNetwork:

    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        #Скрытыае слои
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        #Выходное значение
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1