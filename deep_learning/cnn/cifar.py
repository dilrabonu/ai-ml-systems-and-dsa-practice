"""
================================================================================
  CIFAR-10 — RANGLI RASMLARDAGI OBYEKTLARNI TANISH (To'liq loyiha)
================================================================================
  Bu loyiha 10 turdagi obyektni rangli rasmdan tanийdi:
  samolyot, avtomobil, qush, mushuk, kiyik, it, qurbaqa, ot, kema, yuk mashinasi
 
  MNIST dan FARQI:
    - Rasmlar RANGLI (3 kanal: qizil, yashil, ko'k) — MNIST kulrang (1 kanal) edi
    - Obyektlar murakkab va xilma-xil (real foto) — MNIST oddiy raqamlar edi
    - Shuning uchun CHUQURROQ tarmoq + BatchNorm + Data Augmentation kerak
 
  Texnologiyalar: PyTorch (CNN) + OpenCV (o'z rasmingizni tanish)
  ================================================================================
"""
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader   # disturb info to batches
from torchvision import datasets, transforms # dataset and transformer pictures
import matplotlib.pyplot as plt 
import numpy as np 

torch.manual_seed(42)  # every time the same sample

# Step 2 Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working device: {device}")

# Step 3 Name of classes
class_name = ['airplane', 'car', 'bird', 'cat', 'deer'
              'dog', 'fog', 'horse', 'ship', 'truck']

# Step 4 Data augmentation

train_transform = transforms.Compose([
          transforms.RandomCrop(32, padding=4),   # random crop the image to 32*32 in random place, so model learn with different positions
          transforms.RandomHorizontalFlip(), # round the image horizontal in 50% (cat looks at left or right - so learn both positions)
          transforms.ToTensor(), # Change the image to Tensor and pixels become 0-1
          #normalize the image with mean and std R, G, B
          transforms.Normalize((0.4914, 0.4822, 0.4465), # R, G, B mean 
                               (0.2470, 0.2435, 0.2616)) # R, G, B std
               
])
# in Test we don't need data augmentation, we don't change the image but we evaluate only
# Only we change to Tensor and normalize
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

# Step 5 Load the dataset
train_dataset = datasets.CIFAR10(
    root=".data",
    train=True,  # only train part (50,000)
    download=True,  # if there is no data it will download
    transform=train_transform  # with augmentation
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False, # test part(10,000)
    download=True,
    transform=test_transform # without augmentation
)

print(f"Train samples: {len(train_dataset)}")  # 50000
print(f"Test samples: {len(test_dataset)}")  # 10000

# Step 6 Dataloader
# batch_size=128 - optimal batch size for CIFAR-10
# num_workers - number of CPU cores to use for data loading
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Step 7 CNN architecture

# We use 3 convolutional blocks
# Every block has:
# - Convolutional layer
# - Batch normalization
# - ReLU activation
# - Max pooling
# - Dropout
# Because they help to prevent overfitting and improve generalization
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    # 1 Block
    self.conv1 = nn.Conv2d(in_channels=3, out_channel=32, kernel_size=3, padding=1)
    # BatchNorm2d: after every conv layer to normalize the output
    self.bn1 = nn.BatchNorm2d(32)
    # 2 Block
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    # 3 Block
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)

    # Pooling 
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(128 * 4 * 4, 256)
    self.fc2 = nn.Linear(256, 10) # 10 classes
    
    self.relu = nn.ReLU()

    # Dropout
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    # in : (batch, 3, 32, 32) - color images
    # every block: conv -> BatchNorm -> ReLU -> pool
    # Order is important: conv, normalize, activation, pool
    x = self.pool(self.relu(self.bn1(self.conv1(x))))  # (batch, 32, 16, 16)
    x = self.pool(self.relu(self.bn2(self.conv2(x))))  # (batch, 64, 8, 8)
    x = self.pool(self.relu(self.bn3(self.conv3(x))))  # (batch, 128, 4, 4)

    x = x.view(x.size(0), -1)

    x = self.relu(self.fc1(x)) #(batch, 256)
    x = self.dropout(x)  # overfitting prevention
    x = self.fc2(x)

    return x

model = CNN().to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Step 8 Loss, Optimizer, Scheduler
loss_fn = nn.CrossEntropyLoss()  # for many classes

# Adam optimizer, weight_decay, 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Scheduler - reduce learning rate when loss stops improving
# At first fast learn, at the end slow learn
# StepLR: reduce LR by factor of 0.5 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Step 9 Training 

def training(model, loader, loss_fn, optimizer, device):
    model.train()    # training process(dropout, batchnorm are active)
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 5 step training process
        prediction = model(images)  # prediction
        loss = loss_fn(prediction, labels) # loss calculation
        optimizer.zero_grad() # reset gradients
        loss.backward() # calculate gradients
        optimizer.step() # update weights

        # Statistics
        total_loss += loss.item()
        correct += (prediction.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total

# Step 10 Evaluation
def evaluation(model, loader, device):
    model.eval()   # evaluation (dropout and batchnorm switch off)
    correct = 0
    total = 0
    with torch.no_grad(): # gradient do not need to calculate - fast and a little memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            


