

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mnist import MNIST
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from tkinter import messagebox



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

    def backpropagation(self,X,y_train,learning_rate,lambda_): # Faz BackPropagation para ajustar os parametros
        y_hat = self.forward(X)

        if self.layers[-1].type == "linear": # Starts the Loss Function depending on the output layer
            dA = msederivative(y_hat,y_train)
        elif self.layers[-1].type == "softmax":
            dA = crossentropy_derivative(y_hat,y_train) # Calculates Activation Derivative
        for layer in reversed(self.layers): # Starts backpropagation from output layer
            dA = layer.backward(dA,learning_rate,lambda_)

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

        elif self.type == "softmax":
            z_max = np.max(self.z, axis=1, keepdims=True)
            exp_z = np.exp(self.z - z_max)

            # Calcula a soma das exponenciais para cada linha
            sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)

            # Calcula a ativação Softmax
            activation = exp_z / sum_exp_z

            self.activation = activation
            return self.activation # Using Vectors for better calcs


    def backward(self,dA,l,lambda_):  # Derivative of Loss Function, dA = Loss Derivative
        m = self.inputs.shape[0] # M is equal to the nummber of inputs the Layer receives
        if self.type == "linear":
            self.dZ = dA
        elif self.type == "relu":
            self.aD = relu_derivative(self.z) # Sets Activation Derivative
            self.dZ = self.aD * dA # Multiplies Loss Function by Activation Derivative to get Z Derivative
        elif self.type == "softmax":
            self.dZ = dA

        self.dW = np.dot(self.dZ.T, self.inputs)# Calculates Weight Derivative
        self.dB = np.sum(self.dZ,axis=0,keepdims=True)  # Sums all the dZ values over the Columns, them average them to get b gradients
        regularization = (lambda_ /m) * self.weights # Applies Regularization
        self.dW += regularization
        self.weights -= (self.dW * l) / m
        self.bias -= (self.dB * l) / m
        dA_prev = np.dot(self.dZ,self.weights) # Calculates the new loss derivative to pass to the next layers
        return dA_prev # Returns the derivative for the next layer to use


    # Ouptut Layer will be linear so this will calculate the Output layer derivative
def relu(x):
    return np.maximum(0,x)# Uses and Return the Rectified Linear Value


def msederivative(y_hat,y): # Calculates MSE derivative
    m = len(y_hat) # Size of the Samples
    diff = (y_hat - y) * 2
    return diff

def relu_derivative(z): # Calculates RELU Derivative
    return np.where(z > 0, 1, 0)

def mse(y_hat,y): # Calculates MSE Loss Function
    loss = np.sum((y_hat - y) ** 2) / len(y_hat)
    return loss

def crossentropy_derivative(y_hat,y):
    epsilon = 1e-12  # Para evitar divisão por zero
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    return (y_hat - y) / y.shape[0]


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

def training(nn: NeuralNetwork,X,y_train,learning_rate,epochs,lambda_,batch_size = 32, ): # Trains and test the neural network
    num_samples = X.shape[0] # Num Samples is the number of rows of X
    losses = []
    iterations = []
    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation] # Shuffles teh Data
        y_shuffled = y_train[permutation] # Shuffles the Data

        for i in range(0,num_samples,batch_size): # Goes over the number of batch (the step size) in the examples
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i+ batch_size]
            nn.backpropagation(X_batch, y_batch, learning_rate,lambda_)
        if epoch % 5 == 0:
            # Collect weights from all layers
            weights_list = [layer.weights for layer in nn.layers]

            # Compute predictions on the training set
            y_pred = nn.forward(X)

            # Calculate total loss with regularization over all layers
            total_loss = crossentropy_with_regularization(y_pred, y_train, weights_list, lambda_)
            losses.append(total_loss)
            iterations.append(epoch)
            print(f"Epoch {epoch}, Loss: {total_loss}")



def one_hot_encode(labels,num_classes): # One Hot Encode The Variables for Softman
    return np.eye(num_classes)[labels]

def crossentropy_with_regularization(y_hat, y, weights, lambda_): # Crossentropy Loss with Regularization to Graph it
    m = y.shape[0]
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    cross_entropy_loss = -np.sum(y * np.log(y_hat)) / m
    regularization_loss = (lambda_ / (2 * m)) * sum(np.sum(w ** 2) for w in weights)
    total_loss = cross_entropy_loss + regularization_loss
    return total_loss



mnist = MNIST('mnist_data',gz=True)


# Load the data (returns tuples)
X_train, y_train = mnist.load_training()
X_test, y_test = mnist.load_testing()

# Convert the data to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize the data to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)


layer1 = Layer("relu", 64, 784)
layer2 = Layer("relu", 64, 64)
layer3 = Layer("relu", 64, 64)
layer4 = Layer("softmax",n_neurons=10,n_input = 64)
neuralnetwork = NeuralNetwork([layer1,layer2,layer3,layer4])
training(neuralnetwork,X_train,y_train,0.01,200,0.001)
y_test_labels = np.argmax(y_test, axis=1)
y_train_labels = np.argmax(y_train,axis=1)
softmax_output = neuralnetwork.forward(X_test)
predictions = np.argmax(softmax_output, axis=1)  # Get the index of the highest probability for each sample
accuracy = np.mean(predictions == y_test_labels)   # Calculate the proportion of correct predictions
print(f'Accuracy: {accuracy * 100:.2f}%')

class DrawingApp:
    def __init__(self,neural_network):
        self.window = tk.Tk()
        self.window.title("Desenhe um Número")
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.process_button = tk.Button(self.window, text="Processar", command=self.process_image)
        self.process_button.pack()

        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)
        self.nn = neural_network
        self.clear_button = tk.Button(self.window, text="Limpar", command=self.clear_canvas)
        self.clear_button.pack(pady=10)

    def draw(self,event):
        x, y = event.x, event.y
        r = 5  # Raio do ponto
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        self.draw_image.ellipse([x - r, y - r, x + r, y + r], fill=255, outline=255)

    def process_image(self):
        # Resize to 28x28 pixels
        image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = np.array(image)
        threshold = 50  # You can adjust this value
        image_array = np.where(image_array > threshold, 255, 0)

        coords = np.column_stack(np.where(image_array > 0))
        if coords.size == 0:
            messagebox.showinfo("Erro", "Por favor, desenhe um dígito antes de processar.")
            return
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        cropped_image = image_array[x0:x1 + 1, y0:y1 + 1]
        digit_image = Image.fromarray(cropped_image).resize((20, 20), Image.Resampling.LANCZOS)

        new_image = Image.new('L', (28, 28), 'black')
        new_image.paste(digit_image, (4, 4))  # Center the digit
        image_array = np.array(new_image) / 255.0
        image_input = image_array.flatten().reshape(1, -1)
        output = self.nn.forward(image_input)
        prediction = np.argmax(output, axis=1)[0]
        messagebox.showinfo("Predição", f"Eu acho que é: {prediction}")

    def clear_canvas(self):
        # Limpa todos os desenhos no Canvas do Tkinter
        self.canvas.delete("all")

        # Reinicia a imagem PIL para um fundo preto
        self.image = Image.new("L", (280, 280), "black")
        self.draw_image = ImageDraw.Draw(self.image)

    def run(self):
        self.window.mainloop()


app = DrawingApp(neuralnetwork)
app.run()


