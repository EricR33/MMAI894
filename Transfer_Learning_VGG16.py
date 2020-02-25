# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:47:19 2020

@author: Nicky
"""

#Transfer Learning using VGG16
#Please note we ran it on Google Colab GPU

from google.colab import drive
drive.mount('/content/gdrive')
root_path = 'gdrive/My Drive/Deep Learning/dataset/'
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def train_test_split(datadir, valid_size = .2):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=64),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([
        transforms.Resize(size=128),
        transforms.CenterCrop(size=64),
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

trainloader, testloader = train_test_split(root_path)
print(trainloader.dataset.classes)
print(trainloader.dataset)
# very batch should have 40000*(1-valid_size)/batch_size
len(trainloader)
print(testloader.dataset)
# very batch should have 40000*valid_size/batch_size
len(testloader)
for i, (x, y) in enumerate(trainloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    
for i, (x, y) in enumerate(testloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    
from torchvision import models
model = models.vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
    
model
import torch.nn as nn
model.classifier[6] = nn.Sequential(nn.Linear(4096, 2),                   
                      nn.LogSoftmax(dim=1))

model.classifier

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Move to gpu
model = model.to('cuda')
# Distribute across 2 gpus
model = nn.DataParallel(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_epochs = 10

for epoch in range(n_epochs):
  for data, target in trainloader:
    # Generate predictions
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    out = model(data)
    # Calculate loss
    loss = criterion(out, target)
    # Backpropagation
    loss.backward()
    # Update model parameters
    optimizer.step()
    
prediction_train = []
target_train = []
for data, targets in trainloader:
    data = data.to(device)
    targets = targets.to(device)
    log_ps = model(data)
    # Convert to probabilities
    ps = torch.exp(log_ps).cpu()
    prob = list(ps.detach().numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_train.append(predictions)
    target_train.append(targets)
    
accuracy = []
precision = []
recall = []
for i in range(len(prediction_train)):
    accuracy.append(accuracy_score(target_train[i].cpu(),prediction_train[i]))
    precision.append(precision_score(target_train[i].cpu(),prediction_train[i]))
    recall.append(recall_score(target_train[i].cpu(),prediction_train[i]))
    
print('training accuracy: \t', np.average(accuracy))
print('training precision: \t', np.average(precision))
print('training recall: \t', np.average(recall))

prediction_test = []
target_test = []
for data1, target in testloader:
    data1 = data1.to(device)
    target = target.to(device)
    log_ps = model(data1)
    # Convert to probabilities
    ps = torch.exp(log_ps).cpu()
    prob = list(ps.detach().numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(target)
    
accuracy_test = []
precision_test = []
recall_test = []
for i in range(len(prediction_test)):
    accuracy_test.append(accuracy_score(target_test[i].cpu(),prediction_test[i]))
    precision_test.append(precision_score(target_test[i].cpu(),prediction_test[i]))
    recall_test.append(recall_score(target_test[i].cpu(),prediction_test[i]))
    
print('testing accuracy: \t', np.average(accuracy_test))
print('testing precision: \t', np.average(precision_test))
print('testing recall: \t', np.average(recall_test))