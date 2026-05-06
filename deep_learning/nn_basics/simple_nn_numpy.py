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

def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out

# Network
class TinyNN:
    def __init__(self, n_in: int, n_h: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((n_in, n_h)).astype(np.float32) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_h, dtype=np.float32)
        self.W2 = rng.standard_normal((n_h, n_out)).astype(np.float32) * np.sqrt(2.0 / n_h)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self._cache: dict = {}
    # forward
    def forward(self, X: np.ndarray) -> np.ndarray:
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        P = sigmoid(Z2)
        self._cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P}
        return P
    # Loss
    @staticmethod
    def bce_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # backward
    def backward(self, y: np.ndarray) -> dict:
        c = self._cache
        N = y.shape[0]

        dZ2 = (c["P"] - y) / N
        dW2 = c["A1"].T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_grad(c["Z1"])
        dW1 = c["X"].T @ dZ1
        db1 = dZ1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # SGD
    def step(self, grads: dict, lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    # inference helper
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= threshold).astype(np.int32)

    # training loop
    def train_numpy(X_train, y_train, X_test, y_test,
                    hidden: int = 16, epochs: int = 1000, lr: float = 0.1,
                    verbose: bool = True) -> tuple[TinyNN, list[float]]:
        net = TinyNN(n_in=X_train.shape[1], n_h=hidden, n_out=1)
        losses = []
        for epoch in range(epochs):
            p = net.forward(X_train)
            loss = TinyNN.bce_loss(p, y_train)
            grads = net.backward(y_train)
            net.step(grads, lr)
            losses.append(loss)

            if verbose and epoch % 100 == 0:
                test_acc = (net.predict(X_test) == y_test).mean()
                print(f"epoch {epoch:4d} | loss {loss:.4f} | test_acc {test_acc:.3f}")

        return net, losses
    
        