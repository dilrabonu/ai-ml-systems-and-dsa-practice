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
