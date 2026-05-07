import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset

class TinyNNTorch(nn.Module):
    def __init__(self, n_in: int = 2, n_h: int = 16, n_out: int =1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_h)  #W1, b1 inside
        self.fc2 = nn.Linear(n_h, n_out)  # W2, b2 inside
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

def train_torch(X_train, y_train, X_test, y_test,
                hidden: int = 16, epochs: int = 1000, lr: float = 0.1,
                batch_size: int = 64, seed: int = 0,
                verbose: bool = True):
    torch.manual_seed(seed)
    # Convert numpy -> torch (.from_numpy is zero-copy when possible)
    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xte = torch.from_numpy(X_test)
    yte = torch.from_numpy(y_test)

    model = TinyNNTorch(n_in=X_train.shape[1], n_h=hidden, n_out=1)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    loader = DataLoader(TensorDataset(Xtr, ytr),
                        batch_size=batch_size, shuffle=True)   