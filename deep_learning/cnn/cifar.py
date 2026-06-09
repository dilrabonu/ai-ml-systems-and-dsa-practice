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
from torch.utils.data import DataLoader
