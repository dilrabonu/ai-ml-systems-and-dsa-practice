# PyTorch
import torch 
import torch.nn as nn
import numpy as np 

class SimpleNet(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        self.net == nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Example data
X = torch.tensor([
    [25., 50000., 3., 1.],
    [45., 80000., 10., 0.],
    [30., 60000., 5., 1.]
], dtype=torch.float32)

y = torch.tensor([[0.], [1.], [0.], [1.]])

model = SimpleNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for _ in range(200):
    preds = model(X)
    loss = criterion(preds, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def accuracy(model, X, y):
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        return (preds == y).float().mean().item()

baseline_acc = accuracy(model, X, y)
print("Baseline accuracy:", baseline_acc)

#  Permutation importance
feature_names = ["age", "income", "experience", "owns_house"]
importances = []

for j in range(X.shape[1]):
    X_prem = X.clone()
    idx = torch.randprem(X.shape[0])
    X_prem[:, j] = X_prem[idx, j]
    perm_acc = accuracy(model, X_prem, y)
    importance = baseline_acc - perm_acc
    importances.append(importance)

for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")