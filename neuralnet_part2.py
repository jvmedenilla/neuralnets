# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.in_size = in_size          # (100, 3, 32, 32)
        self.out_size = out_size        # 4
        self.lrate = lrate              # 0.01
        self.hid1_size = 200            # best combo: 150-30 
        self.hid2_size = 130
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        #self.conv1 = nn.Conv2d(3,10, (5,5))  # (in_channels, out_channels, kernel_size(int,int)) <=> (100, 6, 16,16)
        #self.pool = nn.MaxPool2d(2,2)   # (kernel_size(2,2))
        #self.conv2 = nn.Conv2d(10,25, (5,5))
        #self.batnorm1 = nn.BatchNorm2d(6)
        #self.batnorm2 = nn.BatchNorm2d(16)
        #self.fc1 = nn.Linear(25*5*5, self.hid1_size)
        self.conv1 = nn.Conv2d(3,12, (5,5))  # (in_channels, out_channels, kernel_size(int,int)) <=> (100, 6, 16,16)
        self.pool = nn.MaxPool2d(2,2)   # (kernel_size(2,2))
        self.conv2 = nn.Conv2d(12,30, (5,5))
        self.batnorm1 = nn.BatchNorm2d(12)
        self.batnorm2 = nn.BatchNorm2d(30)
        self.fc1 = nn.Linear(30*5*5, self.hid1_size)
        self.fc2 = nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.out_size)
        self.optim = torch.optim.Adam(self.parameters(), self.lrate)

        #self.model = torch.nn.Sequential(self.fc_in, self.relu, self.fc_out)  # 3072x200 

        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #print(x.torch.Tensor.view())
        x = torch.reshape(x, (x.shape[0],3,32,32))
        x = self.batnorm1(self.pool(self.tanh(self.conv1(x))))    # tanh
        x = self.batnorm2(self.pool(self.tanh(self.conv2(x))))
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach().cpu().numpy()

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #print(len(train_set))
    loss_list = []
    out_size = 4
    input_size = train_set.shape[1]
    mean = torch.mean(train_set)
    sd = torch.std(train_set)
    std_train_set = (train_set - mean)/sd
    loss_fn1 = torch.nn.CrossEntropyLoss()
    X = get_dataset_from_arrays(std_train_set, train_labels)
    dataloader = DataLoader(X, batch_size, shuffle=False)   # feed dataset into dataloader object
    net = NeuralNet(0.01, loss_fn1, input_size, out_size)
    #standardize data (d_min, )
    for epoch in range(epochs):
        # for loop to iterate batches in dataloader
        for data in dataloader:
            loss = net.step(data['features'], data['labels'])
        loss_list.append(loss)
    # dev part
    dev_loss = []
    output = net(dev_set)
    for values in output:
        dev_loss.append(np.int(torch.argmax(values)))

    return loss_list,np.array(dev_loss),net
