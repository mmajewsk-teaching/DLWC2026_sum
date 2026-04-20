# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Multi-Layer Perceptron from Scratch
# This notebook extends our work by building a two-layer neural network using only NumPy.

# %%
import numpy as np
import matplotlib.pyplot as plt

# # Part 1: Revisiting Linear Regression with MSE Optimization
# Let's first revisit the MSE-optimized regression from Lab4

# %%
# Load Boston housing dataset
from sklearn.datasets import fetch_openml
import pandas as pd

# @TODO fetch boston dataset with correct parameters
boston = fetch_openml(name=..., version=..., as_frame=...)
df = ...

plt.figure(figsize=(5, 3))
# @TODO plot the data
plt.scatter(..., ...)
plt.xlabel('LSTAT (% lower status of the population)')
plt.ylabel('MEDV (Median value of homes in $1000s)')
plt.title('Boston Housing Dataset: LSTAT vs MEDV')
plt.grid(True)

# %%
# Define a neuron function for linear regression
# @TODO implement the neuron function
def neuron(x, w):
    return ...  

# %%
# Assign initial weights
# @TODO set weights
w = np.array([...])
print(f"Initial weights: {w}")

# %%
# Prepare data with bias term
# @TODO create matrix X with feature and bias term
X = np.vstack([..., ...]).T
# @TODO get target values
y = ...


# MSE Optimization with Stochastic Gradient Descent
current_weights = w.copy()
learning_rate = 0.0001
epochs = 100
mse_history = []
# Run stochastic gradient descent
for epoch in range(epochs):
    # Compute predictions and MSE for tracking
    # @TODO compute predictions using your neuron function
    y_pred = ...
    # @TODO compute error
    error = ...
    # @TODO compute MSE
    mse = ...
    mse_history.append(mse)
    
    # Stochastic updates - process one point at a time
    for j in range(len(df["LSTAT"])):
        # Get a single data point
        # @TODO get single data point with bias
        x_j = ...
        y_j = ...
        
        # Compute prediction and gradient for this point
        # @TODO compute prediction for this point
        y_pred_j = ...
        # @TODO compute gradient 
        gradient_j = ...
        
        # Update weights immediately
        # @TODO update weights using gradient descent
        current_weights = ...


# Plot MSE convergence
plt.figure(figsize=(8, 5))
# @TODO plot mse history
plt.plot(..., ..., '...')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Convergence During Training')
plt.grid(True)

# %%
# Final optimized model
# @TODO compute final predictions
y_pred_final = ...
# @TODO compute final MSE
final_mse = ...

plt.figure(figsize=(8, 5))
plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.7, label='Data')
# @TODO plot the final prediction line
plt.plot(..., ..., '...', label=f'Optimized Model (MSE: {final_mse:.2f})')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('Linear Regression with MSE Optimization')
plt.legend()
plt.grid(True)

# # Part 2: Building a Two-Layer Neural Network

# Define a two-layer neural network
# First layer: 2 neurons
# Second layer: 2 neurons

# @TODO extract features and target (reshape to 2D)
X = df['...'].values.reshape(-1, 1)
y = df['...'].values.reshape(-1, 1)

# Initialize parameters
input_size = 1
# @TODO set hidden size to 2 neurons
hidden_size = ...
output_size = 1

# @TODO initialize weights and biases for first layer
theta1 = np.random.randn(..., ...) * ...
bias1 = np.zeros((1, ...))
# @TODO initialize weights and biases for second layer
theta2 = np.random.randn(..., ...) * ...
bias2 = np.zeros((1, ...))

# Training parameters
alpha = 0.0001  # Learning rate
epochs = 10
m = len(X)

mse_history_nn = []

# Training loop
for epoch in range(epochs):
    for i in range(m):
        # @TODO select random sample
        rand_index = ...
        # @TODO reshape sample for correct dimensions
        x_i = ...
        y_i = ...
        
        # Forward pass
        # @TODO compute input to hidden layer
        hidden_input = ...
        # @TODO compute hidden layer output
        hidden_output = ...  
        # @TODO compute input to output layer
        final_input = ...
        # @TODO compute final output
        final_output = ...
        
        # Compute error
        # @TODO compute error
        error = ...
        
        # Backpropagation
        # @TODO compute gradient for output layer
        d_final = ...
        # @TODO compute gradient for theta2
        d_theta2 = ...
        # @TODO compute gradient for bias2
        d_bias2 = ...
        
        # @TODO compute gradient for hidden layer
        d_hidden = ...
        # @TODO compute gradient for theta1
        d_theta1 = ...
        # @TODO compute gradient for bias1
        d_bias1 = ...
        
        # Parameter updates
        # @TODO update theta2
        theta2 -= ...
        # @TODO update bias2
        bias2 -= ...
        # @TODO update theta1
        theta1 -= ...
        # @TODO update bias1
        bias1 -= ...

    # @TODO compute MSE over the full dataset
    mse = ...
    mse_history_nn.append(mse)

# Plot MSE convergence
plt.figure(figsize=(8, 5))
# @TODO plot MSE history
plt.plot(..., ..., '...')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Convergence During Training')
plt.grid(True)

# Generate predictions
# @TODO compute hidden layer activations
hidden_layer = ...
# @TODO compute predictions
predictions = ...

# Plot results
# @TODO plot data and predictions
plt.scatter(..., ..., color='blue', label='Actual data')
plt.scatter(..., ..., color='red', label='Neural Network Predictions')
plt.xlabel('LSTAT (normalized)')
plt.ylabel('MEDV (normalized)')
plt.legend()
plt.show()

# Now the same thing but using a more object-oriented approach
# Implementing a simple class-based neural network similar to PyTorch style

# %%
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        # @TODO initialize weights and biases with small random values
        self.W1 = ...
        self.b1 = ...
        self.W2 = ...
        self.b2 = ...
    
    def forward(self, x):
        # Forward pass
        # @TODO compute input to first layer
        self.z1 = ...
        # @TODO apply ReLU activation
        self.a1 = ...
        # @TODO compute output layer input
        self.z2 = ...
        return self.z2
    
    def backward(self, x, y, learning_rate):
        # Backward pass
        batch_size = x.shape[0]
        
        # Compute gradients
        # @TODO compute error
        error = ...
        # @TODO compute gradient for W2
        dW2 = ...
        # @TODO compute gradient for b2
        db2 = ...
        
        # @TODO compute gradient for hidden layer
        d_hidden = ...
        # @TODO apply ReLU derivative
        d_hidden[...] = 0
        
        # @TODO compute gradient for W1
        dW1 = ...
        # @TODO compute gradient for b1
        db1 = ...
        
        # Update parameters
        # @TODO update W2 and b2
        self.W2 -= ...
        self.b2 -= ...
        # @TODO update W1 and b1
        self.W1 -= ...
        self.b1 -= ...

# %%
# Prepare normalized data for the neural network
# @TODO extract features and reshape
X_data = ...
# @TODO extract target and reshape
y_data = ...

# Normalize data
# @TODO compute mean and std for normalization
X_mean, X_std = ..., ...
y_mean, y_std = ..., ...

# @TODO normalize data
X_norm = ...
y_norm = ...

# Create and train the model
# @TODO create model with input_size=1, hidden_size=2, output_size=1
model = TwoLayerNet(..., ..., ...)
learning_rate = 0.01
epochs = 1000
batch_size = len(X_norm)  # Use all data at once (batch gradient descent)

# Training loop
for epoch in range(epochs):
    # Forward pass
    # @TODO get model predictions
    outputs = ...
    
    # Compute loss
    # @TODO compute MSE loss
    loss = ...
    
    # Backward pass and update
    # @TODO perform backward pass and update weights
    model.backward(..., ..., ...)
    
    # Print progress occasionally
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Generate predictions
# @TODO get model predictions
predictions_oo = ...
# @TODO denormalize predictions
predictions_oo = ...

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.7, label='Data')
plt.plot(df['LSTAT'], y_pred_final, 'g-', label='Linear Model')
# Sort for smooth line
# @TODO sort indices for smooth plot
sort_idx = ...
# @TODO plot neural network predictions
plt.plot(..., ..., '...', label='OO Neural Network')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('OO-Style Two-Layer Neural Network vs Linear Model')
plt.legend()
plt.grid(True)



# %%
# Now implementing the same network using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Define model architecture
class TorchTwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # @TODO initialize parent class
        super(..., self).__init__()
        # @TODO create first linear layer
        self.layer1 = ...
        # @TODO create ReLU activation
        self.relu = ...
        # @TODO create second linear layer
        self.layer2 = ...
    
    def forward(self, x):
        # Forward pass through the network
        # @TODO apply first layer
        x = ...
        # @TODO apply ReLU activation
        x = ...
        # @TODO apply second layer
        x = ...
        return x

# Convert data to PyTorch tensors
# @TODO convert X_norm to torch tensor
X_tensor = ...
# @TODO convert y_norm to torch tensor
y_tensor = ...

# Create model, loss function, and optimizer
# @TODO create torch model with hidden_size=2
torch_model = TorchTwoLayerNet(..., ..., ...)
# @TODO create MSE loss criterion
criterion = ...
# @TODO create SGD optimizer
optimizer = ...

# Training loop
for epoch in range(1000):
    # Forward pass
    # @TODO compute model outputs
    outputs = ...
    
    # Compute loss
    # @TODO compute loss between outputs and targets
    loss = ...
    
    # Zero gradients, backward pass, and update
    # @TODO clear existing gradients
    ...
    # @TODO compute gradients
    ...
    # @TODO update weights
    ...
    
    # Print progress occasionally
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Generate predictions
# @TODO set model to evaluation mode
torch_model.eval()
with torch.no_grad():
    # @TODO get predictions from model
    predictions_torch = ...
    # @TODO denormalize predictions
    predictions_torch = ...

# Plot all three models
plt.figure(figsize=(8, 5))
plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.5, label='Data')
plt.plot(df['LSTAT'], y_pred_final, 'g-', linewidth=2, label='Linear Model')
# Sort for smooth lines
sort_idx = np.argsort(df['LSTAT'].values)
plt.plot(df['LSTAT'].values[sort_idx], predictions_oo[sort_idx], 'r--', linewidth=2, label='NumPy NN')
# @TODO plot torch model predictions
plt.plot(..., ..., '...', linewidth=2, label='PyTorch NN')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('Comparing Different Neural Network Implementations')
plt.legend()
plt.grid(True)
