import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random as rnd

# Vou fazer duas classes dessa vez, tres classes com uma classe de Neuronio complicou muito.

class NeuralNetwork:
    def __init__(self,layers: list):
        self.layers = layers

    def forward(self,X):
        activation = X
        for i in self.layers:
            activation = i.forward(activation)
        return activation

    def backpropagation(self,X,y_train,learning_rate): # Faz BackPropagation para ajustar os parametros
        y_hat = self.forward(X)
        dA = msederivative(y_hat,y_train)
        for layer in reversed(self.layers): # Starts backpropagation from output layer
            dA = layer.backward(dA,learning_rate)

class Layer:
    def __init__(self,type,n_neurons,n_input): # n_neurons = Numero de Neuronios, n_input = Numero de Inputs
        self.size = n_neurons
        self.type = type
        self.weights = np.zeros((n_neurons,n_input))
        self.bias = np.zeros(n_neurons)
        for row_idx, row in enumerate(self.weights): # Initialize Random Weights
            for col_idx, element in enumerate(row):
                stdev = np.sqrt(2 / n_input)
                self.weights[row_idx,col_idx] = np.random.normal(0,stdev)

    def forward(self,x):
        self.bias = self.bias.reshape(1, -1)
        self.z = np.dot(x,self.weights.T) + self.bias
        self.inputs = x # Save the Inputs of The layer

        if self.type == "relu":
            self.activation = relu(self.z)
            return self.activation

        elif self.type == "linear":
            self.activation = self.z
            return self.activation

    def backward(self,dA,l):  # Derivative of Loss Function, dA = Loss Derivative
        m = self.inputs.shape[0] # M is equal to the nummber of inputs the Layer receives
        if self.type == "linear":
            self.dZ = dA
        elif self.type == "relu":
            self.aD = relu_derivative(self.z) # Sets Activation Derivative
            self.dZ = self.aD * dA # Multiplies Loss Function by Activation Derivative to get Z Derivative

        self.dW = np.dot(self.dZ.T, self.inputs)# Calculates Weight Derivative
        self.dW /= m
        self.dB = np.sum(self.dZ,axis=0,keepdims=True) / m # Sums all the dZ values over the Columns, them average them to get b gradients
        self.weights -= self.dW * l
        self.bias -= self.dB.squeeze() * l
        dA_prev = np.dot(self.dZ,self.weights) # Calculates the new loss derivative to pass to the next layers
        return dA_prev # Returns the derivative for the next layer to use



    # Ouptut Layer will be linear so this will calculate the Output layer derivative


def relu(x):
    return np.maximum(0,x)# Uses and Return the Rectified Linear Value


def msederivative(y_hat,y): # Calculates MSE derivative
    m = len(y_hat) # Size of the Samples
    diff = (y_hat - y) * (2 / m)
    return diff

def relu_derivative(z): # Calculates RELU Derivative
    return np.where(z > 0, 1, 0)

def mse(y_hat,y): # Calculates MSE Loss Function
    loss = np.sum((y_hat - y) ** 2) / len(y_hat)
    return loss





def forwardtest(): # Testing if Forwarding is done correctly
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5]])
    layer1 = Layer("relu", 3, 2)
    layer2 = Layer("linear", 1, 3)
    layer1.weights = np.array([[0.1, 0.2],  # Neuron 1
                               [0.3, 0.4],  # Neuron 2
                               [0.5, 0.6]])
    layer1.bias = np.array([[0.1, 0.2, 0.3]])
    layer2.weights = np.array([[0.7, 0.8, 0.9]])
    layer2.bias = np.array([[0.1]])
    nn = NeuralNetwork([layer1, layer2])
    return nn.forward(X)  # Expected Values: 3.36,5.12,6.88,8.64


np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)  # For linear dataset
noise = np.random.normal(0, 1, X.shape)
y = 3 * X + 5 + noise

# Define the neural network architecture
layer1 = Layer("relu", 10, 1)  # 1 input feature, 10 neurons
layer2 = Layer("linear", 1, 10)  # Output layer

# Initialize the neural network
nn = NeuralNetwork([layer1, layer2])
losses = []
iterations = []
for i in range(1000):
    nn.backpropagation(X,y,0.01)
    prediction = nn.forward(X)
    losses.append(mse(prediction,y))
    iterations.append(i)

print(nn.forward(X))
print(" ")
print(y)
plt.plot(iterations,losses)
plt.show()