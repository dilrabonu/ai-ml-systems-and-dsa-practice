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

# Step 3
class_name = ['airplane', 'car', 'bird', 'cat', 'deer'
              'dog', 'fog', 'horse', 'ship', 'truck']