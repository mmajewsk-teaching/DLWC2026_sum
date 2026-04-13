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

# # Linear Regression Revisited
# This notebook revisits linear regression from first principles using numpy.

import numpy as np
import matplotlib.pyplot as plt

# # Understanding Linear Regression
# We'll explore why we need linear regression and how to implement it from scratch

from sklearn.datasets import fetch_openml
import pandas as pd

# %%
# @TODO we are doing linear regression
def neuron(x, w):
    return ...

# ## Part 1 - Using simple equations

# %%
# @TODO get the boston dataset
boston = fetch_openml(name=..., version=1, as_frame=True)
df = ....frame

# %%
# @TODO plot LSTAT and MEDV variables
plt.figure(figsize=(5, 3))
plt.scatter(df[...], df[...])
plt.xlabel('LSTAT (% lower status of the population)')
plt.ylabel('MEDV (Median value of homes in $1000s)')
plt.title('Boston Housing Dataset: LSTAT vs MEDV')
# @TODO pick the first point from the dataset
x_picked, y_picked = ..., ...
plt.scatter(x_picked, y_picked, color="black", s=100)
plt.grid(True)

# %%
# @TODO add random weights
w = np.array([..., ...])

# %%
# @TODO assign correct values, use x from dataset
x_1 = ...
x_b = np....(x_1)
x = np....([x_1, x_b]).T
y_pred = neuron(x, w)
error = (df["MEDV"].values - y_pred)
mse = np.mean(error**2)

# %%
plt.figure(figsize=(5, 3))
# @TODO plot the boston data
plt.scatter(..., ...)
# @TODO plot predicted line
plt.plot(..., ..., c='r', label='Initial Line')
# @TODO plot picked point in black
plt.scatter(..., ..., color=..., s=100, label='Focus Point')
plt.title(f'MSE: {mse:.2f}')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)

# The correct_weights function calculates weights to make a line pass through a point (x_n, y_n).
#
# For a linear model with equation: y = w_0*x + w_1
# We want to find weights such that: y_n = w_0*x_n + w_1
#
# Using LaTeX notation:
# $y_n = w_0 \cdot x_n + w_1$
#
# To find w_0, we rearrange:
# $w_0 = ...TODO$
#
# This is exactly what our function calculates:
# w_0 = ...TODO
#
# For w_1, we use our new w_0 value:
# $w_1 = ...TODO$
#
# This ensures that our line equation y = w_0*x + w_1 will pass through the point (x_n, y_n).
# Since we have 2 unknowns (w_0 and w_1) but only 1 constraint (the line must pass through one point),
# there are infinitely many solutions. Our approach gives one particular solution.

# %%
# @TODO fill in the function
def correct_weights(x_n, y_n, w):
    w_0 = ...
    w_1 = ...
    return w_0, w_1

# %%
x_n = df['LSTAT'][0]
y_n = df['MEDV'][0]
x_n, y_n

# %%
w_0, w_1 = correct_weights(x_n, y_n, w)
exact_weights = np.array([w_0, w_1])

print(f"Focus point: ({x_n}, {y_n})")
print(f"Original weights: {w}")
print(f"Exact weights: {exact_weights}")

# %%
plt.figure(figsize=(5, 3))
plt.scatter(df['LSTAT'], df['MEDV'])
# @TODO
x_with_bias = np.vstack([..., np.ones_like(...)]).T
# @TODO fill in the weights that we have just calculated
y_pred_exact = neuron(x_with_bias, ...)
plt.plot(x_1, y_pred_exact, 'g-', label='Exact Line')
plt.scatter(x_n, y_n, color="black", s=100, label='Focus Point')
plt.title('Line passing through focus point')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)

# ## Part 2 - Applying the equation iteratively
#
# Now lets implement a way to not update the weights straight away, do it iteratively.
# In the plot below you should see how the line slowly approaches our "ideal" weights.

# %%
current_weights = w.copy()
alpha = 0.2
y_pred_init = neuron(x, current_weights)

weight_history = []
for i in range(9):
    # @TODO Calculate the correct weights for picked point, based on current_weights
    w_0_ideal, w_1_ideal = ...(..., ..., ...)
    ideal_weights = np.array([w_0_ideal, w_1_ideal])

    # @TODO Calculate the difference between ideal_weights and current weights
    weight_diff = ideal_weights - ...
    # @TODO update current weights based on the fraction of the weight difference
    current_weights = ... + alpha * weight_diff
    weight_history.append(current_weights.copy())

# %%
fig, axe = plt.subplots(3, 3, figsize=(12, 8))
for i, wh in enumerate(weight_history):
    y_pred = neuron(x, wh)
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1, y_pred_init, 'r--', alpha=0.5, label='Initial Line')
    axe[i//3, i%3].plot(x_1, y_pred, 'g-', label=f'Iteration {i+1}')
    axe[i//3, i%3].set_title(f'Iteration {i+1}')
plt.tight_layout()

# # Regression to multiple points

# ## Part 3 - Simple regression to all the points
#
# Now we would like to actually do the regression but to all of the points.
# We will use MSE as a metric for our method.

# %%
current_weights = w.copy()
alpha = 0.02

mse_history = []
epoch_weights = []
for i in range(9):
    for j in range(len(df["LSTAT"])):
        # @TODO each iteration use next point
        x_n = df[...][...]
        y_n = df[...][...]
        # @TODO calculate ideal weights
        w_0_ideal, w_1_ideal = ...(..., ..., ...)
        ideal_weights = np.array([w_0_ideal, w_1_ideal])
        # @TODO Calculate the difference between ideal_weights and current weights
        weight_diff = ideal_weights - ...
        # @TODO update current weights based on the fraction of the weight difference
        current_weights = ... + alpha * weight_diff

        # @TODO calculate prediction
        y_pred = neuron(x, ...)
        error = df["MEDV"].values - y_pred
        mse = np.mean(error**2)
        mse_history.append(mse)

    epoch_weights.append(current_weights.copy())

# %%
fig, axe = plt.subplots(3, 3, figsize=(12, 8))
for i, wh in enumerate(epoch_weights):
    y_pred = neuron(x, wh)
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1, y_pred_init, 'r--', alpha=0.5, label='Initial Line')
    axe[i//3, i%3].plot(x_1, y_pred, 'g-', label=f'Iteration {i+1}')
    axe[i//3, i%3].set_title(f'Iteration {i+1}')
plt.tight_layout()

# %%
plt.figure(figsize=(8, 5))
# @TODO plot mse history
plt.plot(..., 'ro-', markersize=1)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE over Iterations - Instability in Single-Point Updates')
plt.grid(True)

# The MSE history shows instability as we're updating weights after each point.
# Each point pulls the weights in a different direction — the line oscillates.

# ## Part 4 - Full-batch MSE gradient descent
#
# Instead of the analytical correction, we now optimise MSE directly using its gradient.

# Mean Squared Error (MSE) is defined as the average of squared differences between predictions and actual values.
# Given:
# - $X$ is our input matrix with features (LSTAT) and bias term
# - $w$ is our weight vector (slope and intercept)
# - $y$ is our target values (MEDV)
# - $\hat{y} = Xw$ is our prediction
#
# The MSE is calculated as:
# $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - X_i w)^2$
#
# To optimize MSE, we need its gradient with respect to weights $w$:
# $\nabla_w \text{MSE} = \frac{\partial}{\partial w}\frac{1}{n}\sum_{i=1}^{n}(y_i - X_i w)^2$
#
# Using the chain rule and $\frac{\partial}{\partial w}(y_i - X_i w) = -X_i^T$:
# $\nabla_w \text{MSE} = -\frac{2}{n}X^T(y - Xw)$
#
# The gradient descent update rule:
# $w_{new} = w_{old} - \alpha \nabla_w \text{MSE}$

# %%
# @TODO implement the MSE gradient function
def mse_gradient(X, y, w):
    y_pred = ...
    return ...

# %%
X = np.vstack([df['LSTAT'], np.ones(len(df['LSTAT']))]).T
y = df['MEDV'].values

# %%
current_weights = w.copy()
learning_rate = 0.0001
epochs = 500
mse_history_fullbatch = []
plot_epochs = set(np.linspace(0, epochs - 1, 9, dtype=int))
saved_weights_fb = {}

for epoch in range(epochs):
    # @TODO compute gradient and update weights
    gradient = ...
    current_weights = current_weights - ... * ...
    y_pred = neuron(X, current_weights)
    mse_val = np.mean((y - y_pred)**2)
    mse_history_fullbatch.append(mse_val)
    if epoch in plot_epochs:
        saved_weights_fb[epoch] = current_weights.copy()

w_fullbatch = current_weights.copy()

# %%
sort_idx = np.argsort(x_1)
fig, axe = plt.subplots(3, 3, figsize=(12, 8))
for i, ep in enumerate(sorted(saved_weights_fb)):
    y_pred = neuron(X, saved_weights_fb[ep])
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1[sort_idx], y_pred[sort_idx], 'b-', label=f'Epoch {ep+1}')
    axe[i//3, i%3].set_title(f'MSE: {mse_history_fullbatch[ep]:.2f} (Epoch {ep+1})')
    axe[i//3, i%3].grid(True)
fig.suptitle('Full-Batch Gradient Descent', fontsize=16)
plt.tight_layout()

# %%
plt.figure(figsize=(8, 5))
plt.plot(mse_history_fullbatch, 'b-')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Loss Curve - Full-Batch Gradient Descent')
plt.grid(True)

# ## Part 5 - Mini-batch gradient descent
#
# Full-batch uses all points per update (stable but slow per step).
# Per-point updates (Part 3) are fast but noisy.
# **Mini-batch** is the practical compromise: we split data into small chunks (e.g. 32)
# and compute the gradient on each chunk.

# %%
current_weights = w.copy()
learning_rate = 0.0001
epochs = 500
batch_size = 32
mse_history_minibatch = []
plot_epochs_mb = set(np.linspace(0, epochs - 1, 9, dtype=int))
saved_weights_mb = {}

np.random.seed(42)
for epoch in range(epochs):
    # @TODO shuffle the data each epoch
    indices = np.random...(len(y))
    X_shuffled = X[...]
    y_shuffled = y[...]

    # @TODO iterate over mini-batches and update weights
    for start in range(0, len(y), batch_size):
        X_batch = X_shuffled[start:start + ...]
        y_batch = y_shuffled[start:start + ...]
        gradient = ...
        current_weights = current_weights - ... * ...

    y_pred = neuron(X, current_weights)
    mse_val = np.mean((y - y_pred)**2)
    mse_history_minibatch.append(mse_val)
    if epoch in plot_epochs_mb:
        saved_weights_mb[epoch] = current_weights.copy()

w_minibatch = current_weights.copy()

# %%
fig, axe = plt.subplots(3, 3, figsize=(12, 8))
for i, ep in enumerate(sorted(saved_weights_mb)):
    y_pred = neuron(X, saved_weights_mb[ep])
    axe[i//3, i%3].scatter(df['LSTAT'], df['MEDV'], alpha=0.5)
    axe[i//3, i%3].plot(x_1[sort_idx], y_pred[sort_idx], 'g-', label=f'Epoch {ep+1}')
    axe[i//3, i%3].set_title(f'MSE: {mse_history_minibatch[ep]:.2f} (Epoch {ep+1})')
    axe[i//3, i%3].grid(True)
fig.suptitle('Mini-Batch Gradient Descent (batch_size=32)', fontsize=16)
plt.tight_layout()

# %%
plt.figure(figsize=(8, 5))
plt.plot(mse_history_fullbatch, 'b-', label='Full-Batch')
plt.plot(mse_history_minibatch, 'g-', label='Mini-Batch')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Loss Curve Comparison: Full-Batch vs Mini-Batch')
plt.legend()
plt.grid(True)

# %%
print(f"Full-batch final MSE: {mse_history_fullbatch[-1]:.2f}")
print(f"Mini-batch final MSE: {mse_history_minibatch[-1]:.2f}")
