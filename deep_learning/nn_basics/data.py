from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np 

def load_moons(n_samples: int=1000, noise: float = 0.2, seed: int = 42):
    """Returns (X_train, X_test, y_train, y_test) all as float32 numpy arrays."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=seed)