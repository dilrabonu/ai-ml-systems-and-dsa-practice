import torch
import torch.nn as nn
import torch.optim as optim 

X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])
y = torch.tensor([[0.], [1.], [1.], [0.]])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

model = MLP()

loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training Loop
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(X))