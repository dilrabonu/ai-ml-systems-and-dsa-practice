import torch 
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = SimpleClassifier(in_dim=768, hidden=256, n_classes=10)


