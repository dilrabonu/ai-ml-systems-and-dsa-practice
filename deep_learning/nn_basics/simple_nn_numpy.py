"""
Two-layer neural network from scratch in pure NumPy.

Architecture: Input(2) -> Linear -> RaLU -> Linear -> Sigmoid (binary)
Loss : Binary cross-entropy
Optimizer: Vanilla SGD on the full batch
"""
from __future__ import annotations
import numpy as np

# activation
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def sigmoid