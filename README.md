# Neural Network from Scratch in Python

This repository contains a custom implementation of a feedforward neural network built from scratch in Python. The project is designed for educational purposes to gain a deeper understanding of neural networks by implementing core functionalities manually, without relying on high-level libraries like TensorFlow or PyTorch.

## Recent Updates

- **16/10/2024**: Created the initial neural network structure with support for ReLU activation and a linear output layer. This foundational work significantly enhanced understanding of neural network mechanics.
- **14/11/2024**: Implemented the Adam optimization algorithm, which resolved previous training issues. This improvement boosted the model's accuracy on the MNIST dataset from **92% to 98%** with fewer epochs and without data augmentation.

## Features

- **Customizable Neural Network Architecture**: Easily define network layers and activation functions to experiment with different architectures.
  - **Supported Activation Functions**:
    - **ReLU (Rectified Linear Unit)**
    - **Softmax** (for multi-class classification)
- **Loss Functions**:
  - **Mean Squared Error (MSE)**
  - **Cross-Entropy Loss** (with support for regularization)
- **Optimization Algorithms**:
  - **Adam Optimizer**: Incorporates adaptive learning rates and momentum, leading to faster convergence and improved performance.
- **Backpropagation Implementation**: Manually coded backpropagation algorithm for training the network using gradient descent optimization.
- **Regularization Techniques**:
  - **L2 Regularization**: Helps prevent overfitting by penalizing large weights.
- **Batch Training**: Supports mini-batch gradient descent with adjustable batch sizes.
- **Evaluation Metrics**:
  - **Accuracy Calculation**
  - **Loss Tracking and Visualization**
