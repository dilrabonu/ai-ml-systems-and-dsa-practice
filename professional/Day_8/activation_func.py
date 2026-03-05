import torch
import torch.nn as nn

def step(x: torch.Tensor) -> torch.Tensor:
    return (x >= 0).to(x.dtype)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))

def tanh(x: torch.Tensor) -> torch.Tensor:
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)) # zero centred

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0.0)

def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.where(x >= 0, x, negative_slope * x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_stable = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x_stable)
    return exp / exp.sum(dim=dim, keepdim=True)

# Forward pass
class TinyNet(nn.Module):
    """
    A small neural net:
    input -> Linear -> Ralu -> Linear -> sigmoid or softmax
    This shows the forward pass explicitly (no nn.ReLU / nn.Sigmoid used).
    """
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        # Linear layers learn weights and biases
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor, task: str ='binary') -> torch.Tensor:
        z1 = self.fc1(x) # first affine transform : z1 = xW + b 
        a1= relu(z1) # non-linearity: allows learning complex patterns
        logits = self.fc2(a1) # second affine transform z2 = a1W + b

        if task == 'binary':
            probs = sigmoid(logits)
            return probs
        elif task == 'multiclass':
            probs = softmax(logits, dim=1)
            return probs
        else:
            return logits

# Quick usage demo
torch.manual_seed(0)

X = torch.randn(5, 3)

# Binary classifier
bin_model = TinyNet(in_features=3, hidden=8, out_features=1)
bin_probs = bin_model(X, task='binary') # forward pass
print("Binary probabilities shape": bin_probs.shape)
print(bin_probs)

# Multi-class classifier
mul_model = TinyNet(in_features=3, hidden=8, out_features=4)
mul_probs = mul_model(X, task='multiclass')
print("\nMulticlass probabilities shape": mul_probs.shape)
print(mul_probs)
print("Row sums (should be 1):", mul_probs.sum(dim=1))