import torch 
import torch.nn as nn

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        
