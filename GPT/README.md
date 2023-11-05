# GPT Implementation with PyTorch

This repository contains an implementation of the GPT (Generative Pre-trained Transformer) model using the PyTorch library. GPT is designed to pre-train deep generative models using a transformer architecture.

## Implemented Classes

The following classes have been implemented:

| Class                | Description   |
|----------------------|---------------|
| `Embeddings`         | Handles token and position embeddings, normalization, and dropout. |
| `AttentionHead`      | Represents a single attention head, applying linear transformations and scaled dot product attention. |
| `MultiHeadAttention` | Contains multiple attention heads and concatenates their outputs. |
| `FeedForward`        | Implements the feed-forward neural network present in each transformer block. |
| `Decoder`            | Represents a single transformer block, consisting of multi-head attention and feed-forward neural network. |
| `myGPT`              | Represents the entire GPT architecture, encapsulating all the above components into a cohesive model. |

## Unit Tests

Unit tests for the implemented classes are available in the `unit_test` directory.
