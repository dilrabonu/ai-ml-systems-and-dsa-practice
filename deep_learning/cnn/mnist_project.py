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



