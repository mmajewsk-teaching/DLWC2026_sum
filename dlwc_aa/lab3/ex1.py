#!/usr/bin/env python
# coding: utf-8

# # Lab 1: Building a Basic RAG System
# 
# In this lab, we'll create a simple Retrieval Augmented Generation (RAG) system using PyTorch and Hugging Face models.

# ## Setup
# First, let's import the necessary libraries and set up our environment.

# In[31]:


# Standard library imports
import os
import time

# In[32]:


# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

# In[33]:


# Check if CUDA is available and set device
device = torch.device("cuda" if ... else "cpu")
print(f"Using device: {device}")

# ## Download and Load Language Model
# 
# We'll use a pre-trained language model from Hugging Face for generating embeddings.

# In[34]:


# Define which model to use - we'll use a small but effective model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"We'll use the model: {model_name}")

# In[35]:


# Load the tokenizer - this converts text to tokens the model can understand
tokenizer = .......(..., cache_dir="/tmp")
print(f"Tokenizer loaded with vocabulary size: {len(tokenizer)}")

# In[36]:


# Let's see how the tokenizer works with a simple example
example_text = "Hello, this is a sample text for our RAG system!"
tokens = tokenizer(example_text)
print("Input text:", example_text)
print("Token IDs:", tokens["input_ids"])
print("Decoded tokens:", tokenizer....)

# In[37]:


# Now load the actual model
model = .......(..., cache_dir="/tmp").to(device)
print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")

# ## Generate Embeddings
# 
# Now we'll see how to generate embeddings for text. Embeddings are vector representations that capture semantic meaning.

# In[38]:


# First, prepare text for the model by tokenizing it
text_to_embed = "This is a sample text to demonstrate embedding generation."
encoded_input = ...(..., padding=True, truncation=True, return_tensors="pt").to(device)
print("Encoded input shape:")
for key, value in encoded_input.items():
    print(f"  {key}: {value.shape}")

# In[39]:


# Pass the encoded input through the model
with torch.no_grad():
    model_output = ...(...)

# Look at the model output
print("Model output keys:", model_output.keys())
print("Last hidden state shape:", model_output.last_hidden_state.shape)
# This is a 3D tensor: [batch_size, sequence_length, hidden_size]

# In[40]:


# To get a single vector for the entire text, we'll use mean pooling
# Since we're only processing a single sentence without batching,
# we can simply take the mean of the token embeddings
token_embeddings = model_output.last_hidden_state

# In[41]:


# Calculate mean across the sequence dimension (dim=1)
final_embedding = torch....(token_embeddings, dim=1).squeeze()
print("Final embedding shape:", final_embedding.shape)

# Convert to numpy array for easier handling
embedding = final_embedding.cpu().numpy()
print("Embedding numpy shape:", embedding.shape)
print("First 5 values:", embedding[:5])

# In[ ]:



