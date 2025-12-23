# MNIST Handwritten Digit Classification

A simple neural network implementation from scratch using NumPy for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements a 2-layer neural network for digit recognition without using deep learning frameworks. The model achieves classification of digits (0-9) using only NumPy for matrix operations.

## Model Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 16 neurons with Sigmoid activation
- **Output Layer**: 10 neurons with Softmax activation (one per digit class)

## Features

- **Custom Neural Network Implementation**: Built from scratch without TensorFlow/PyTorch
- **Activation Functions**:
  - Sigmoid for hidden layer
  - Softmax for output layer
- **Training Algorithm**: Stochastic Gradient Descent (SGD)
- **Backpropagation**: Manual implementation of gradient computation
- **Data Preprocessing**: Pixel normalization (0-255 → 0-1)

## Dataset Structure

The project expects the following files:
- `train.csv` - Training data with labels
- `mnist_test.csv` - Test data for evaluation

Data format: First column contains labels (0-9), remaining 784 columns contain pixel values.

## Implementation Details

### Key Functions

- `init_params()` - Initialize weights and biases randomly
- `forward_prop()` - Forward propagation through the network
- `back_prop()` - Backpropagation to compute gradients
- `update_params()` - Update parameters using gradient descent
- `SGD()` - Main training loop using stochastic gradient descent
- `evaluate()` - Test model accuracy on validation set

### Hyperparameters

- **Learning Rate (α)**: 0.9
- **Hidden Layer Size**: 16 neurons
- **Training Epochs**: 50
- **Development Set**: First 1000 samples
- **Training Set**: Remaining samples

## Usage

1. Ensure you have the required data files (`train.csv`, `mnist_test.csv`)
2. Open `model.ipynb` in Jupyter Notebook
3. Run all cells sequentially

```python
# Train the model
w1, b1, w2, b2 = SGD(x_train, y_train, epochs=50)

# Evaluate performance
evaluate(w1, b1, w2, b2, x_train, y_train)
```

## Requirements

```
numpy
pandas
```

Install dependencies:
```bash
pip install numpy pandas
```

## Model Training

The model uses Stochastic Gradient Descent where:
- Each training example is processed individually
- Weights are updated after every single example
- Training accuracy is printed after each epoch

## Results

The model will output:
- Training accuracy after each epoch
- Final development set accuracy

## Project Structure

```
mnist/
├── model.ipynb          # Main notebook with model implementation
├── train.csv            # Training dataset
├── mnist_test.csv       # Test dataset
└── README.md            # This file
```

## Notes

- Data is normalized by dividing pixel values by 255
- One-hot encoding is used for labels
- The implementation uses matrix operations for efficient computation
- Stability improvements applied (e.g., in softmax calculation)

## Future Improvements

- Implement batch or mini-batch gradient descent
- Add more hidden layers
- Experiment with different activation functions (ReLU, etc.)
- Add regularization techniques (L2, dropout)
- Save and load trained model parameters
- Visualize predictions and misclassifications

## License

Educational project for learning neural network fundamentals.
