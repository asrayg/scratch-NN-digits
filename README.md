# From Scratch Neural Network Digit Recognizer

This project is a simple digit recognition system implemented using a neural network. The neural network is trained on the MNIST dataset, which consists of handwritten digit images and their corresponding labels.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Mathematical Explanation](#mathematical-explanation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project demonstrates how to build a neural network from scratch to recognize handwritten digits. The network is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of digits from 0 to 9.

## Dataset
The MNIST dataset is used for training and evaluating the model. The dataset is split into a training set and a development (validation) set.

## Installation
To run the notebook, you need to have Python and several libraries installed. You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib jupyter
```

## Usage
To use this project, simply run the provided Jupyter notebook. The notebook includes all the necessary code to load the data, train the neural network, and evaluate its performance.

```bash
jupyter notebook digit-recogniser.ipynb
```

## Model Architecture
The neural network consists of the following layers:
- Input layer: 784 neurons (28x28 pixels)
- Hidden layer: 10 neurons
- Output layer: 10 neurons (one for each digit 0-9)

## Mathematical Explanation

### Initialization
Weights and biases are initialized randomly:
```python
W1 = np.random.rand(10, 784) - 0.5
b1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(10, 10) - 0.5
b2 = np.random.rand(10, 1) - 0.5
```

### Forward Propagation
The forward propagation involves computing the activations of each layer:

1. **Input Layer to Hidden Layer**:
    \[
    Z1 = W1 \cdot X + b1
    \]
    \[
    A1 = \text{ReLU}(Z1)
    \]

    - **ReLU (Rectified Linear Unit) Activation Function**:
        \[
        \text{ReLU}(z) = \max(0, z)
        \]
      The ReLU function introduces non-linearity into the model, allowing it to learn more complex patterns.

2. **Hidden Layer to Output Layer**:
    \[
    Z2 = W2 \cdot A1 + b2
    \]
    \[
    A2 = \text{softmax}(Z2)
    \]

    - **Softmax Activation Function**:
        \[
        \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
        \]
      The softmax function converts the raw scores (logits) from the output layer into probabilities that sum to 100%. This is crucial for classification tasks where we need to interpret the output as probabilities of each class.

### Loss Function
The loss function used is the **Cross-Entropy Loss**, which measures the difference between the true labels and the predicted probabilities:
\[
\text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{10} y_{ij} \log(A2_{ij})
\]
Where:
- \( m \) is the number of examples.
- \( y_{ij} \) is the true label for example \( i \) and class \( j \).
- \( A2_{ij} \) is the predicted probability for example \( i \) and class \( j \).

### Backward Propagation
The backward propagation involves computing the gradients and updating the weights and biases:

1. **Output Layer to Hidden Layer**:
    \[
    dZ2 = A2 - Y
    \]
    \[
    dW2 = \frac{1}{m} dZ2 \cdot A1^T
    \]
    \[
    db2 = \frac{1}{m} \sum dZ2
    \]

2. **Hidden Layer to Input Layer**:
    \[
    dZ1 = W2^T \cdot dZ2 \cdot \text{ReLU}'(Z1)
    \]
    \[
    dW1 = \frac{1}{m} dZ1 \cdot X^T
    \]
    \[
    db1 = \frac{1}{m} \sum dZ1
    \]

    - **Derivative of ReLU Function**:
        \[
        \text{ReLU}'(z) = \begin{cases} 
        1 & \text{if } z > 0 \\
        0 & \text{if } z \leq 0 
        \end{cases}
        \]

### Gradient Descent Optimization
The weights and biases are updated using gradient descent:
\[
W1 = W1 - \alpha dW1
\]
\[
b1 = b1 - \alpha db1
\]
\[
W2 = W2 - \alpha dW2
\]
\[
b2 = b2 - \alpha db2
\]

Where:
- \( \alpha \) is the learning rate, a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.

## Results
The performance of the model can be evaluated using the accuracy on the development set. The notebook includes code to compute and display the accuracy.
