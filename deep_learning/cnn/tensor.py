import torch

# Create
a = torch.tensor([1.0, 2.0, 3.0])  # vector
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # matrix (2*2)

print(a.shape)
print(b.shape)

print(a + 1) # 1 adds for each elements
print(a * 2) # 2 multiplies for each elements
print(b @ b) # matrix multiplication

zeros = torch.zeros(2, 3) # 2*3 zero matrix
random = torch.randn(2, 3)  # 2*3 random numbers
