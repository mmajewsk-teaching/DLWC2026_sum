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

# # Linear Regression on Real Data
#
# We'll fit a linear model to real housing data using **gradient descent**.
#
# **MSE** (Mean Squared Error) measures how far our predictions are from the truth — on average, squared.
# The **gradient** of MSE tells us which direction to nudge the weights to reduce the error.
# **Gradient descent** repeatedly takes small steps in the negative gradient direction until the loss converges.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd

# %%
def neuron2(x, w):
    return x@w

# ## Loading the Dataset

# %%
boston = fetch_openml(name="boston", version=1, as_frame=True)

# %%
df = boston.frame

# %%
df

# ## Visualizing the Data

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['LSTAT'], df['MEDV'])
plt.xlabel('LSTAT')
plt.ylabel('MEDV')

# ## Fitting a Line Manually

# %%
x_1 = df["LSTAT"].values
x_b = np.ones_like(x_1)
X_train = np.vstack([x_1, x_b]).T
y_train = df['MEDV'].values

# %%
w = np.array([..., ...])
y_pred = neuron2(X_train, w)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df['LSTAT'], df['MEDV'])
plt.plot(x_1, y_pred, c='r')

# ## Mean Squared Error
# MSE = (1/N) Σ (y_true - y_pred)². Smaller is better.

# %%
def mse(y_actual, y_predicted):
    return np.mean(...)

# %%
def l_mse(X, y, w):
    y_pred = neuron2(X, w)
    return ...

# %%
l_mse(X_train, y_train, w)

# ## Comparing Models

# %%
w2 = np.array([0.22925899, 1.12375])
w3 = np.array([3.22925899, 0.12375])

w2_loss = l_mse(X_train, y_train, w2)
print(f"Model 1 Loss: {w2_loss:.2f}")

w3_loss = l_mse(X_train, y_train, w3)
print(f"Model 2 Loss: {w3_loss:.2f}")

# ## MSE Gradient
# The gradient tells us the direction of steepest increase in loss.
# We move **opposite** to the gradient to decrease loss.

# %%
def mse_gradient(X, y, w):
    y_pred = neuron2(X, w)
    gradient = -(2/len(y)) * ... @ (...)
    return gradient

# %%
mse_gradient(X_train, y_train, w2)

# %%
mse_gradient(X_train, y_train, w3)

# ## Gradient Descent

# +
# %%
# +
plt.figure(figsize=(10, 6))
w = np.array([4.22925899, 6.12375])
plt.scatter(df['LSTAT'], df['MEDV'])

iterations = 140
learning_rate = 0.0001
losses = []
for _ in range(iterations):
    gradient = ...
    loss = ...
    losses.append(loss)
    w = w - ... * ...
    print(w, loss)

y_pred = neuron2(X_train, w)
mse_val = mse(y_train, y_pred)
plt.title(f"MSE: {mse_val:.2f}")
plt.plot(x_1, y_pred, c='r')
# -

# ## Loss Curve
# Plotting MSE over iterations shows how the model converges.

# %%
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Loss curve over gradient descent iterations")
plt.grid(True)

# ## Train/Test Split
# So far we trained and evaluated on the same data. To measure **generalization**,
# we split the data into train and test sets.

# %%
from sklearn.model_selection import train_test_split

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %%
w = np.array([4.22925899, 6.12375])
train_losses = []
test_losses = []
for _ in range(iterations):
    gradient = mse_gradient(X_tr, y_tr, w)
    w = w - ... * ...
    train_losses.append(l_mse(X_tr, y_tr, w))
    test_losses.append(l_mse(X_te, y_te, w))

# %%
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Train vs Test loss")
plt.legend()
plt.grid(True)

# %%
print(f"Final train MSE: {train_losses[-1]:.2f}")
print(f"Final test MSE:  {test_losses[-1]:.2f}")
