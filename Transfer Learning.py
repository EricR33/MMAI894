# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:12:46 2020

@author: Nicky
"""

from torchvision import transforms   
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models




data_dir = 'C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/'
def train_test_split(datadir, valid_size = .2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transform)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader
trainloader, testloader = train_test_split(data_dir)
print(trainloader.dataset.classes)
print(trainloader.dataset)
# very batch should have 40000*(1-valid_size)/batch_size
len(trainloader)
print(testloader.dataset)
# very batch should have 40000*valid_size/batch_size
len(testloader)

from torchvision import models
model = models.resnet50(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
    
model
# Add on classifier
model.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=2
    ),
    torch.nn.Sigmoid()
)

model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

n_epochs = 2

for epoch in range(n_epochs):
  for data, target in trainloader:
    # Generate predictions
    out = model(data)
    # Calculate loss
    loss = criterion(out, target)
    # Backpropagation
    loss.backward()
    # Update model parameters
    optimizer.step()