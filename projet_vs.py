# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:41:14 2022

@author: Victo
"""
# -*- coding: utf-8 -*-
"""
Deep Learning @ unistra
LeNet-5 CNN architecture
@author: Stefano Bianchini

""" 

import sys
sys.path.append("..") 

import torch
import torch.nn as nn
import torchvision
import  torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
 


#%% *** Setting / Dataset ***

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the hyper-parameters of the network
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
 
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
 
test_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=False,
                                          transform=transforms.ToTensor())
 
# Wrap into the data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Let's look at some examples
print('The size of the input (train) image is: ', next(iter(train_dataset))[0].shape)
print ('The label of the input (train) image is ', next(iter(train_dataset))[1])


 

#%% *** LeNet-5 CNN architecture ***

# - Layer C1 is a convolution layer with 6 convolution kernels of 5x5 and the size of feature mapping is 28x28;

# - Layer S2 is the subsampling/pooling layer that outputs 6 feature graphs of size 14x14. 
#   Each cell in each feature map is connected to 2x2 neighborhoods in the corresponding feature map in C1;

# - Layer C3 is a convolution layer with 16 5-5 convolution kernels; 

# - Layer S4 is similar to S2, with size of 2x2 and output of 16 5x5 feature graphs;

# - Layer C5 is a convolution layer with 120 convolution kernels of size 5x5.

# - FC1 and FC2 are fully connected layers


class LetNet5(nn.Module):
    def __init__(self, num_clases=10):
        super(LetNet5, self).__init__()
 
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
 
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.c5 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
 
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
 
        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes)
        )
 
    def forward(self, x):
        out = self.c1(x)
        out = self.c3(out)
        out = self.c5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
 
 
model = LetNet5(num_classes).to(device)
 


#%% *** LeNet at work !! ***

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 

# Train the network
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
 
        # Quick remind:                                              
        # 1. Pass the images to the model                                               
        # 2. Compute the loss using the output and the labels                          
        # 3. Compute gradients and update the model using the optimizer 
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
 
images.shape
labels.shape
labels
model(images).shape

# Test the accuracy
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
 
torch.save(model.state_dict(), 'LetNet-5.ckpt')



    ######################################################################################################
    # PLAY ALONE                                                                                         #
    #                                                                                                    #
    # Experiment with different:                                                                         #
    #    - number of hidden layers                                                                       #
    #    - size of the filters                                                                           #
    #    - activation functions                                                                          #
    #    - type of pooling                                                                               #
    #                                                                                                    #
    #                                                                                                    #
    # Try performing data augmentation https://pytorch.org/docs/stable/torchvision/transforms.html       #
    # Can you improve further the performance of the model?                                              #
    #                                                                                                    #
    #                                                                                                    #
    # Use the LeNet-5 architecture to identify if the number in the image is greater than or equal to 5  #
    ######################################################################################################

