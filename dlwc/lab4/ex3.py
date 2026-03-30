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

# # Linear Regression with a Neuron
#
# In the previous exercises our neuron used a **step function** (threshold) to produce 0 or 1.
# If we remove the activation entirely, the neuron computes `y = x · w` — a **linear model**.
# This is the foundation of linear regression.
#
# Later we'll add non-linear activations (sigmoid, ReLU) back in to see how they shape the output.

# ## Level 1: Creating a Linear Neuron
# No activation — just a raw dot product.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def neuron2(x, w):
    return ...

# ## Level 2: Generating Linear Data
# Creating a simple dataset to visualize linear models

# %%
x_1 = np.linspace(0, 10, 25)

# %%
x_2 = np.zeros_like(x_1) + 1

# %%
X = np.vstack((x_1, x_2)).T

# %%
X[:5]

# ## Level 3: Creating Linear Models
# Different weights create different linear functions

# %%
w1 = np.array([0.5, 1])
Y1 = neuron2(X, w1)

# %%
w2 = np.array([0.3, 1])
Y2 = neuron2(X, w2)

# %%
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], Y1, label="Line 1: slope=0.5, intercept=1")
plt.plot(X[:, 0], Y2, label="Line 2: slope=0.3, intercept=1")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# ## Level 4: Activation Functions
# Real neural networks use non-linear activations after the weighted sum.
# Two of the most common:
# - **Sigmoid**: squashes output to (0, 1) — useful for probabilities
# - **ReLU**: outputs max(0, z) — simple, fast, and the default in most modern networks

# %%
def sigmoid(z):
    return 1 / (1 + ...)

def relu(z):
    return np....(0, z)

# %%
z = np.linspace(-6, 6, 200)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid")
plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)

plt.subplot(1, 2, 2)
plt.plot(z, relu(z))
plt.title("ReLU")
plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()

# ## Level 5: Neuron Output with Different Activations
# Let's see how each activation transforms the output of our linear neuron.

# %%
w = np.array([0.5, 1])
z_raw = neuron2(X, w)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(X[:, 0], z_raw)
plt.title("No activation (linear)")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(X[:, 0], sigmoid(z_raw))
plt.title("Sigmoid activation")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(X[:, 0], relu(z_raw))
plt.title("ReLU activation")
plt.grid(True)
plt.tight_layout()
