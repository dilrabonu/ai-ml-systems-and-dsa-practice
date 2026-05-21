import torch
import torch.nn as nn 

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)  # 4 - in, 8- out
        self.linear2 = nn.Linear(8, 1)  # 8 - in, 1- out

    def forward(self, x):
        x = self.linear1(x)  # pass linear 1
        x = torch.relu(x)    #ReLU activation (gradient flow)
        x = self.linear2(x)   # pass linear 2
        return x

model = MyModule()
print(model)