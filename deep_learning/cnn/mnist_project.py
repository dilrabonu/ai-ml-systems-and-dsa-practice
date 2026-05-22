"""
Aim: Classify MNIST pictures with hand written digits
Model train the picture and tell which digit is it?

Dataset:
- 60000 training images
- 10000 test images
- 28x28 pixels
- 1 channel (grayscale)

Steps:
1. Import Libraries
2. Choose Device GPU/CPU
3. Settings Hyperparameters
4. Load Data
5. Create Model
6. Loss function and optimizer
7. Train and evaluate
8. The main train step epoch
9. Save Model
10. Test Model and Predict

# Install PyTorch and torchvision
pip install torch torchvision
python mnist_project.py

"""
# 1 Step Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 2 Step Choose Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# 3 Step Setting Hyperparameters
settings = {
    "batch_size": 64,   # Number of images to process at once
    "learning_rate": 0.001, # Learning rate for the optimizer
    "epochs": 5,    # Number of epochs to train the model
    "hidden_size": 128, # Number of neurons in the hidden layer
}
print(f"Settings: {settings}")

# 4 Step: Load Data
"""
transform - Transform the data to tensors and normalize it

ToTensor() - Convert the data to tensors - from 0-255 to 0-1
Normalize() - Normalize the data - subtract mean and divide by standard deviation
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Train to learn model
train_dataset = datasets.MNIST(
    root="./data",   # where to save the (data folder)
    train=True,      #training data
    download=True,   #download data if not exist
    transform=transform,  #apply transforms
)

# Test for evaluate the model( this part is not seen by model)
test_dataset = datasets.MNIST(
    root="./data", 
    train=False,    # test data
    download=True,  # download data if not exist
    transform=transform,  # apply transforms
)

print(f"Train : {len(train_dataset)} pictures")
print(f"Test: {len(test_dataset)} pictures")

train_loader = DataLoader(
    train_dataset,
    batch_size=settings['batch_size'],
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=settings["batch_size"],
    shuffle=False,
)

# 5 Step: Create Model
"""
nn.Module - PyTorch base class for all neural networks
3 layers
Input 784 (28*28 pixels = 784)
1 - layer 784 -> 128 (hidden layer)
2 - layer 128 -> 64 (hidden layer)
Output 64 -> 10 (for 10 classes - digits 0-9)
"""
class DigitRecognize(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()  
        # Flatten 28*28 to 784 convert to 1 vector
        self.flatten = nn.Flatten()

        #nn.Linear = matrix multiplication  layer
        # there are output= input @ weight + bias 
        self.layer1 = nn.Linear(28*28, hidden_size)  # 784 -> 128
        self.layer2 = nn.Linear(hidden_size, 64)     # 128 -> 64
        self.layer3 = nn.Linear(64, 10)              # 64 -> 10 (number of classes)

        # ReLU - activation function - negative values to 0, positive not changed max(0, x)
        # This helps model to learn complex patterns
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        forward information will flow inside Model
        This function work automatically when we call input
        """
        x = self.flatten(x) # picture 28*28 to 784 vactor
        x = self.layer1(x)  # 1 layer passed
        x = self.relu(x)    # activation function negative values to 0
        x = self.layer2(x)  # 2 layer passed
        x = self.relu(x)    # activation function
        x = self.layer3(x)  # 3 layer passed and 10 classes
        return x

model = DigitRecognize(hidden_size=settings['hidden_size']).to(device)
print("\nModel structure:")
print(model)

# How many parameters in model
total_param = sum(p.numel() for p in model.parameters())
print(f"\nTotal sums of parameters: {total_param:, }")

# Step 6 Loss function and optimizer

"""
Loss function - measure how wrong model is
- CrossEntropyLoss - for classification
- MSELoss - for regression
- L1Loss - for regression
"""
loss_fn = nn.CrossEntropyLoss()

"""
Optimizer - update weights in model
- SGD - simple optimizer
- Adam - adaptive optimizer it can measure learning rate
- RMSprop - adaptive optimizer
"""
optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"])

# 7 Step: Train and Evaluate functions
def epoch_train(model, loader, loss_fn, optimizer, device):
    """
    During one epoch model learns from data
    Train process with 5 steps:
    1) forward - model makes answer
    2) loss - calculate loss
    3) zero_grad - clean old gradients
    4) backward - measure new gradient
    5) optimizer.step - update weights

"""
model.train()
total_loss = 0
correct_sum = 0
total_samples = 0

for X_batch, y_batch in loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    # here main 5 steps
    pred = model(X_batch)   # forward
    loss = loss_fn(pred, y_batch)  # claculate loss

    optimizer.zer_grad()    # clean old gradient
    loss.backward()         # calculate new gradient
    optimizer.step()        # update weights

    # We gather for statistics
    