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

# # Logical Operations with Neurons
#
# A single neuron draws a **linear decision boundary** in the input space.
# This means it can solve any problem where a straight line (or hyperplane) separates
# the two classes. We'll see which logic gates work — and which don't.

# %%
import numpy as np

# %%
def neuron(x, w):
    return ((...) >= 0)*1

# ## Level 1: Basic Neuron Test
# Testing the neuron with basic inputs

# %%
x = np.array([0, 0, 1])
w = np.array([0, 0, 0])

# %%
neuron(x, w)

# ## Level 2: OR Gate Implementation
# OR is **linearly separable**: only (0,0) maps to 0, the rest to 1.
# A single line can separate these — so a single neuron is enough.

# %%
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [1]
])

# %%
w = np.array([..., ..., ...])

# %%
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# ## Level 3: AND Gate Implementation
# AND is also linearly separable: only (1,1) maps to 1.
# We just need a stricter threshold (higher bias weight).

# %%
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# %%
w = np.array([..., ..., ...])

# %%
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# ## Level 4: XOR Problem

# %%
X = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# %%
w = np.array([..., ..., ...])

# %%
for i in range(4):
    print(f"Input: {X[i][0:2]}, Output: {neuron(X[i], w)}, Expected: {Y[i][0]}")

# ## Level 5: Solving XOR with a 2-Layer Network
# The key insight: XOR = OR AND NAND.
#
# We combine three neurons:
# - Neuron 1: OR gate
# - Neuron 2: NAND gate (NOT AND)
# - Neuron 3: AND gate on the outputs of neurons 1 and 2
#
# This is our first **multi-layer** network!

# %%
w_or = np.array([1, ..., ...])
w_nand = np.array([..., ..., 1.5])
w_and = np.array([..., 1, -1.5])

# %%
for i in range(4):
    x = X[i]
    h1 = neuron(x, ...)
    h2 = neuron(x, ...)
    hidden_out = np.array([h1, h2, 1])
    out = neuron(hidden_out, w_and)
    print(f"Input: {x[0:2]}, OR={h1}, NAND={h2}, XOR={out}, Expected: {Y[i][0]}")
