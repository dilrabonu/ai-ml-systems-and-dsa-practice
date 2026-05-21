import torch

# Create
a = torch.tensor([1.0, 2.0, 3.0])  # vector
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # matrix (2*2)

print(a.shape)
print(b.shape)