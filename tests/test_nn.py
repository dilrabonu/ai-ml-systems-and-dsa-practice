import numpy as np
import pytest
from deep_learning.nn_basics.simple_nn_numpy import TinyNN, sigmoid, relu

def test_sigmoid_extremes_dont_overflow():
    """Stable sigmoid must handle very large |x| without producing nan/inf."""
    x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    out = sigmoid(x)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[2] == pytest.approx(1.0, abs=1e-6)

def test_forward_shapes():
    net = TinyNN(n_in=4, n_h=8, n_out=1)
    X = np.random.randn(32, 4).astype(np.float32)
    p = net.forward(X)
    assert p.shape == (32, 1)
    assert ((p > 0) & (p < 1)).all()       # sigmoid is bounded

def test_loss_decreases_after_step():
    """Sanity: one SGD step should not INCREASE loss for a small enough lr."""
    net = TinyNN(2, 8, 1, seed=0)
    X = np.random.randn(64, 2).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) > 0).astype(np.float32)
    p1 = net.forward(X); l1 = TinyNN.bce_loss(p1, y)
    grads = net.backward(y)
    net.step(grads, lr=0.05)
    p2 = net.forward(X); l2 = TinyNN.bce_loss(p2, y)
    assert l2 <= l1 + 1e-6                 # tolerance for floating point noise

def test_gradient_check_numerical():
    """Gold-standard test: compare analytical gradient to numerical gradient."""
    net = TinyNN(2, 4, 1, seed=1)
    X = np.random.randn(8, 2).astype(np.float32)
    y = np.random.randint(0, 2, (8, 1)).astype(np.float32)

    net.forward(X)
    grads = net.backward(y)

    eps = 1e-4
    for name in ("W1", "b1", "W2", "b2"):
        param = getattr(net, name)
        flat = param.ravel()
        # check 5 random entries (full check is slow)
        idx = np.random.choice(flat.size, size=5, replace=False)
        for i in idx:
            orig = flat[i]
            flat[i] = orig + eps; loss_plus  = TinyNN.bce_loss(net.forward(X), y)
            flat[i] = orig - eps; loss_minus = TinyNN.bce_loss(net.forward(X), y)
            flat[i] = orig
            num_grad = (loss_plus - loss_minus) / (2 * eps)
            ana_grad = grads[name].ravel()[i]
            assert abs(num_grad - ana_grad) < 1e-2, f"{name}[{i}]: num={num_grad}, ana={ana_grad}"