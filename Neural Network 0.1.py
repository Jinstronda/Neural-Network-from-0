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
        self.bias = np.full((1, n_neurons), 0.01)
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
        self.dB = np.sum(self.dZ,axis=0,keepdims=True) / m # Sums all the dZ values over the Columns, them average them to get b gradients
        self.weights -= self.dW * l
        self.bias -= self.dB * l
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


import numpy as np
import matplotlib.pyplot as plt

# Generate the dataset
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
noise = np.random.normal(0, 1, X.shape)
y = 3 * X + 5 + noise

# Normalize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Normalize the target data
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y_normalized = (y - y_mean) / y_std

# Define the neural network architecture
layer1 = Layer("relu", 10, 1)  # 1 input feature, 10 neurons
layer2 = Layer("relu", 10, 10)
layer3 = Layer("linear", 1, 10)  # Output layer


# Initialize the neural network
nn = NeuralNetwork([layer1, layer2,layer3])

# Training parameters
epochs = 5000
learning_rate = 0.001

# Training loop
losses = []
iterations = []
for i in range(epochs):
    nn.backpropagation(X_normalized, y_normalized, learning_rate)
    prediction_normalized = nn.forward(X_normalized)
    loss = mse(prediction_normalized, y_normalized)
    losses.append(loss)
    iterations.append(i)

# Print the final normalized predictions and the normalized target values
print("Final normalized predictions:\n", nn.forward(X_normalized))
print("\nNormalized target values:\n", y_normalized)

# Plot the training loss over epochs
plt.plot(iterations, losses, label="Training Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Predict using the trained model
y_pred_normalized = nn.forward(X_normalized)

# Denormalize the predicted values
y_pred = y_pred_normalized * y_std + y_mean

# Plot the predictions vs. actual data
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Predicted Data")
plt.title("Actual vs Predicted Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
