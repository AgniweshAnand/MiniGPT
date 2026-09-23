# NanoGPT: Character-Level Language Model from Scratch

A PyTorch implementation of an autoregressive Transformer decoder trained on the Tiny Shakespeare dataset, developed following Andrej Karpathy's *Zero to Hero* series.

The project demonstrates the evolution from a simple Bigram statistical baseline to a generative Transformer architecture featuring causal multi-head self-attention, residual connections, and pre-layer normalization.

---

## Features

- **Character-Level Tokenization:** Custom integer mapping for unique characters in the dataset.
- **Decoder-Only Transformer Architecture:**
  - Token and learned Positional Embeddings
  - Multi-Head Causal Self-Attention (masked autoregressive attention)
  - Position-wise Feed-Forward Networks with ReLU non-linearities
  - Residual (Skip) Connections
  - Pre-Layer Normalization (`nn.LayerNorm`)
  - Dropout regularization
- **Training Pipeline:** Mini-batch generation, periodic train/validation loss estimation, and AdamW optimization.
- **Autoregressive Text Generation:** Context cropping to context window (`block_size`) and multinomial sampling.

---

## Model Architecture & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `batch_size` | 16 | Sequences processed in parallel per step |
| `block_size` | 32 | Maximum context length for predictions |
| `max_iters` | 5000 | Total optimization iterations |
| `eval_interval` | 100 | Step frequency for evaluating train/val loss |
| `eval_iters` | 200 | Batches averaged per evaluation |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `n_embd` | 64 | Embedding vector dimension |
| `n_head` | 4 | Number of parallel attention heads (`head_size = 16`) |
| `n_layer` | 4 | Number of stacked Transformer blocks |
| `dropout` | 0.0 | Dropout rate for regularization |

---

## Quickstart

### 1. Requirements

Install PyTorch:

```bash
pip install torch