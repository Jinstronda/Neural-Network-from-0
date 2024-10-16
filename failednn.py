import numpy as np
import pandas as pd
import matplotlib as mpl
import random as rnd

class NeuralNetwork:
    def __init__(self,layers: list):
        self.layers = layers

    def forward(self,x_train):
        activation = x_train
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def train(self,x_train,y_train,l,epochs): # Performs back propagation and trains the neural network.
        x_prediction = self.forward(x_train)
        for i in range(epochs):
            for layer in self.layers:
                if isinstance(layer,OutputLayer):
                    OutputLayer.forward()



class Layer:
    def __init__(self,size,asize): # asize é o tamanho dos inputs da Layer anterior, quando criarmos as layers vamos especificar isso
        self.size = size
        self.neurons = []
        for i in range(size):
            self.neurons.append(Neuron(asize))

    def __str__(self):
        return f"Number of Neurons: {self.size}"

    def forward(self,a):
        for i in self.neurons:
            i.forward(a)
        self.output = np.array([i.activation for i in self.neurons])
        return self.output

    def backward(self,y_train):
        for i in self.neurons:
            i.backward()


class Neuron:
    def __init__(self, asize): # Shape é um TUPLE com (Neuronios da Layer Atual, Numero de Neuronios de entrada)
        stdev = np.sqrt(2/asize) # Calcula desvio Padrão
        self.weight = np.random.normal(0,stdev,size=asize)
        self.bias = 0
        self.weight_gradient = np.zeros_like(self.weight)
        self.bias_gradient = 0

    def __str__(self): # Retorna weight e bias do neuronio, para testes.
        return f"Weight:{self.weight},Bias:{self.bias}"

    def forward(self,a):
        self.z = np.dot(self.weight, a) + self.bias
        self.activation = relu(self.z)

class OutputLayer:
    def __init__(self,size,asize): # asize é o tamanho dos inputs da Layer anterior, quando criarmos as layers vamos especificar isso
        self.size = size
        self.neurons = []
        for i in range(size):
            self.neurons.append(OutputNeuron(asize))

    def forward(self, a):
        for i in self.neurons:
            i.forward(a)
        self.output = np.array([i.activation for i in self.neurons])
        return self.output

    def backward(self, y_train,prev_layer_activation):
        for i in self.neurons:
            i.backward(prev_layer_activation)

class OutputNeuron:
    def __init__(self, asize): # Shape é um TUPLE com (Neuronios da Layer Atual, Numero de Neuronios de entrada)
        stdev = np.sqrt(2/asize) # Calcula desvio Padrão
        self.weight = np.random.normal(0,stdev,size=asize)
        self.bias = 0
        self.weight_gradient = np.zeros_like(self.weight)
        self.bias_gradient = 0

    def forward(self,a):
        self.z = np.dot(self.weight, a) + self.bias
        self.activation = self.z

    def backward(self,y_train,activation):
        errorderivative = mse_derivative(y_train,self.activation)








def relu(x): # Activation Function
    return np.maximum(0,x)

def mean_squared_error(y_true, y_pred):# Mean Squared Error
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred): # Mean Squared Errror Gradient
    return 2 * (y_pred - y_true) / y_true.size

