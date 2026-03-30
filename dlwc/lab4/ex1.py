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

# # Part 1: Understanding the Perceptron
#
# A **perceptron** is the simplest building block of a neural network.
# It takes inputs, multiplies each by a weight, sums them up, and applies a threshold
# to produce a binary output (0 or 1).
#
# We will build this step by step — starting from raw arithmetic, then gradually
# refactoring toward clean, vectorized code. Each level introduces a new idea
# that makes our implementation more general and efficient.

# ## Level 1: Basic Weighted Sum
# We start with the most explicit form: individual variables and manual arithmetic.

# %%
x1, x2, x3 = 0, 2, 0

# %%
w1, w2, w3 = -1, 1, 3

# %%
b = 3

# %%
output = ...*w1 + x2*... + ...*w3
output

# %%
if output ... b:
    result = 1
else:
    result = 0

# %%
result

# ## Level 2: Treating Bias as a Weight
# The threshold `b` is mathematically equivalent to a weight on a constant input of 1.
# By absorbing the bias into the weight vector, we simplify the neuron to a single dot product + sign check.

# %%
x1, x2, x3 = 0, 2, 0
w1, w2, w3 = -1, 1, 3
b = 3

# %%
x1*... + ...*w2 + x3*... - 1*... >= 0

# %%
w4 = ...

# %%
if x1*... + ...*w2 + ...*w3 - ...*w4 >= 0:
    result = 1
else:
    result = 0

# ## Level 3: Using Lists for Inputs and Weights
# Using lists lets us generalize to **any number of inputs** without changing the code structure.

# %%
x = [x1, x2, x3, 1]
w = [w1, w2, w3, w4]

# %%
s = 0
for i in range(len(x)):
    s += ...*...

# %%
(s >= 0)*1

# ## Level 4: Using Zip for Cleaner Code
# `zip` pairs elements from two sequences — no more manual indexing.

# %%
s = 0
for x_n, w_n in ...(x, w):
    s += x_n*w_n

# %%
(s >= 0)*1

# ## Level 5: Using List Comprehension
# A more Pythonic one-liner — the weighted sum becomes a single expression.

# %%
tmp_s = [...*... for x_n, w_n in ...(x, w)]

tmp_s

# %%
s = sum(tmp_s)

# %%
(s >= 0)*1

# ## Level 6: Using NumPy for Vector Operations
# NumPy's `dot` product and `@` operator do the same thing in one fast, vectorized call.
# This is both cleaner and **much faster** for large inputs.

# %%
import numpy as np

# %%
(np....(x, w) >= 0)*1

# %%
xa = np.array(x)
wa = np.array(w)

# %%
xa...wa

# ## Level 7: Creating a Neuron Function
# Finally we wrap everything in a reusable function. This is the perceptron:
# `f(x) = step(x · w)`

# %%
def neuron(x, w):
    return ((...) >= 0)*1
