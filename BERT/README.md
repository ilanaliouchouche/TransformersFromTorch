# BERT Implementation with PyTorch

This repository contains an implementation of the BERT (Bidirectional Encoder Representations from Transformers) model using the PyTorch library. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.

## Implemented Classes

The following classes have been implemented:

| Class                | Description   |
|----------------------|---------------|
| `Embeddings`         | Handles token and position embeddings, normalization, and dropout. |
| `AttentionHead`      | Represents a single attention head, applying linear transformations and scaled dot product attention. |
| `MultiHeadAttention` | Contains multiple attention heads and concatenates their outputs. |
| `FeedForward`        | Implements the feed-forward neural network present in each transformer block. |
| `Encoder`            | Represents a single transformer block, consisting of multi-head attention and feed-forward neural network. |
| `myBert`             | Represents the entire Bert architecture, encapsulating all the above components into a cohesive model. |

## Unit Tests

Unit tests for the implemented classes are available in the `unit_test` directory.
