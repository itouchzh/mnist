# -*- coding: utf-8 -*-
# @Time    : 2022/8/26
# @Author  : rickHan
# @Software: PyCharm
# @File    : mnist.py
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

# use gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# parameter
batch_size = 256
# thread
num_workers = 4
lr = 1e-4
epochs = 20
image_size = 28
data_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

train_data = MNIST('./data', train=True, transform=data_transform, download=False)
test_data = MNIST('./data', train=False, transform=data_transform, download=False)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# draw image
image, label = iter(test_loader).next()
# plt.imshow(image[0][0])
# print(label[0])
#
# design net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv net
        """
        input(256,1,28,28)
        """
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),  # (32, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (32, 12, 12)
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),  # (64, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (64, 4, 4)
            nn.Dropout(0.3)
        )
        # Fully connected network
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


model = Net()
model = model.cuda()
# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainLoss = []


def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    trainLoss.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


accRate = []


def val(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)  # Dimension is 1 (maximum index)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    accRate.append(acc)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))


for epoch in range(1, 21):
    train(epoch)
    val(epoch)

x = []
for i in range(20):
    x.append(i)
plt.plot(x, trainLoss)
plt.xlabel('epoch')
plt.ylabel('train loss')
plt.show()

plt.plot(x, accRate)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
