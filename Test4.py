# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:26:52 2020

@author: Nicky
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'C:/Users/Nicky/Desktop/Old Com/Queens Master/MMAI894/Project/dataset/'
def train_test_split(datadir, valid_size = .2):
    train_transform = transforms.Compose([transforms.Resize(128),
                                       transforms.ToTensor(),
                                       ])
    test_transform = transforms.Compose([transforms.Resize(128),
                                      transforms.ToTensor(),
                                      ])
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

# making sure each batch is roughly balanced
for i, (x, y) in enumerate(trainloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    
for i, (x, y) in enumerate(testloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))

# converting training images into torch format
final_train = np.array(final_train).reshape(28000, 1, 128, 128)
final_train  = torch.from_numpy(final_train)
final_train = final_train.float()

# converting the target into torch format
final_target_train = np.array(final_target_train).astype(int)
final_target_train = torch.from_numpy(final_target_train)

# converting validation images into torch format
val_x = np.array(X_test).reshape(12000, 1, 128, 128)
val_x  = torch.from_numpy(val_x)
val_x = val_x.float()

# converting the target into torch format
val_y = np.array(y_test).astype(int)
val_y = torch.from_numpy(val_y)


# method 1 (not performing well)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,16,4)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(16,32,2)
        self.fc1 = nn.Linear(32 * 30 * 30, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        #print(x.shape)
        x = x.view(-1,32*30*30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for epoch in range(2):  # loop over the dataset multiple times
    
    accuracy = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

## method 2 (perform better)

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

torch.manual_seed(0)

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
            # Defining another 2D convolution layer
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.20),
        )

        self.linear_layers = Sequential(
            Linear(8192, 512),
            ReLU(inplace=True),
            Dropout(),
            Linear(512, 256),
            ReLU(inplace=True),
            Dropout(),
            Linear(256,10),
            ReLU(inplace=True),
            Dropout(),
            Linear(10,2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

print(model)
    
# batch size of the model
batch_size = 64

# number of epochs to train the model
n_epochs = 2

for epoch in range(1, n_epochs+1):

    train_loss = 0.0
        
    permutation = torch.randperm(final_train.size()[0])

    training_loss = []
    for i in tqdm(range(0,final_train.size()[0], batch_size)):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = final_train[indices], final_target_train[indices]
        
        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
    
torch.save(model, 'model.pt')
the_model = torch.load('model.pt')

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

torch.manual_seed(0)
# prediction for training set
prediction = []
target = []
permutation = torch.randperm(final_train.size()[0])
for i in tqdm(range(0,final_train.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = final_train[indices], final_target_train[indices]

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction.append(predictions)
    target.append(batch_y)
    
# training accuracy
accuracy = []
precision = []
recall = []
for i in range(len(prediction)):
    accuracy.append(accuracy_score(target[i].cpu(),prediction[i]))
    precision.append(precision_score(target[i].cpu(),prediction[i]))
    recall.append(recall_score(target[i].cpu(),prediction[i]))
    
print('training accuracy: \t', np.average(accuracy))
print('training precision: \t', np.average(precision))
print('training recall: \t', np.average(recall))


# checking the performance on validation set

#method 1 testing all together
torch.manual_seed(0)
output = model(val_x.cpu())
softmax = torch.exp(output).cpu()
prob = list(softmax.detach().numpy())
predictions = np.argmax(prob, axis=1)
accuracy_score(val_y, predictions)

#method 2 testing it in batches

batch_size = 32

prediction_test = []
target_test = []
permutation_test = torch.randperm(val_x.size()[0])
for i in tqdm(range(0,val_x.size()[0], batch_size)):
    indices = permutation_test[i:i+batch_size]
    batch_x, batch_y = val_x[indices], val_y[indices]

    ##if torch.cuda.is_available():
        ##batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model(batch_x)

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_test.append(predictions)
    target_test.append(batch_y)
    
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