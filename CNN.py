import cv2
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


length = 3072
# 加载测试集
dict_data = []
dict_data.append(unpickle("data_batch_1"))
dict_data.append(unpickle("data_batch_2"))
dict_data.append(unpickle("data_batch_3"))
dict_data.append(unpickle("data_batch_4"))
dict_data.append(unpickle("data_batch_5"))  # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

import torch
import torch.nn as nn


class NeuralNetworks(nn.Module):
    def __init__(self):
        super(NeuralNetworks, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = None

    def forward(self, x):
        logits = self.model(x)
        return logits

    def loss_func(self, Logits, target):
        self.cross_entropy_loss = self.loss_fn(Logits, target)
        return self.cross_entropy_loss

    def backward_func(self):
        self.cross_entropy_loss.backward()


model = NeuralNetworks()

epoches = 50
learning_rate = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
batch_size = 64

import matplotlib.pyplot as plt


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    aver_loss = torch.tensor([0.])
    for batch, (X, y) in enumerate(dataloader):
        pred = model.forward(X)
        loss = model.loss_func(pred, y)

        optimizer.zero_grad()
        model.backward_func()
        optimizer.step()
        temp = torch.tensor(loss.item())
        aver_loss+=temp
        if batch % 70 == 0:
            print(temp)
    print("average loss: {}".format(aver_loss / size))


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    aver_acc = torch.tensor([0.])
    for batch, (X, y) in enumerate(dataloader):
        with torch.no_grad():
            pred = model.forward(X)
            aver_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    print("average accuracy: {}".format(aver_acc / size))


training_dataset0 = TensorDataset(torch.tensor(dict_data[0][b'data'], dtype=torch.float32).view((10000, 3, 32, 32)),
                                 torch.tensor(dict_data[0][b'labels']))
training_dataset1 = TensorDataset(torch.tensor(dict_data[1][b'data'], dtype=torch.float32).view((10000, 3, 32, 32)),
                                 torch.tensor(dict_data[1][b'labels']))
training_dataset2 = TensorDataset(torch.tensor(dict_data[2][b'data'], dtype=torch.float32).view((10000, 3, 32, 32)),
                                 torch.tensor(dict_data[2][b'labels']))
training_dataset3 = TensorDataset(torch.tensor(dict_data[3][b'data'], dtype=torch.float32).view((10000, 3, 32, 32)),
                                 torch.tensor(dict_data[3][b'labels']))
testing_dataset = TensorDataset(torch.tensor(dict_data[4][b'data'], dtype=torch.float32).view((10000, 3, 32, 32)),
                                torch.tensor(dict_data[4][b'labels']))
training_data0 = DataLoader(training_dataset0, batch_size=batch_size, shuffle=True)
training_data1 = DataLoader(training_dataset1, batch_size=batch_size, shuffle=True)
training_data2 = DataLoader(training_dataset2, batch_size=batch_size, shuffle=True)
training_data3 = DataLoader(training_dataset3, batch_size=batch_size, shuffle=True)
testing_data = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

for i in range(epoches):
    print(f"Epoch {i+1}\n------------")
    train_loop(training_data0, model, optimizer)
    train_loop(training_data0, model, optimizer)
    train_loop(training_data0, model, optimizer)
    train_loop(training_data0, model, optimizer)
    test_loop(testing_data, model)
    print("")