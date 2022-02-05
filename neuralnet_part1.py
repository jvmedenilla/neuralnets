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
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from typing import ForwardRef
import numpy as np
from numpy.core.fromnumeric import argmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.in_size = in_size          # 32x32x3 = 3072
        self.out_size = out_size        # 4
        self.hid_size = 150             # can be changed
        self.loss_fn = loss_fn          # defined externally (above)
        self.lrate = lrate              # 0.01
        self.fc_in = nn.Linear(self.in_size, self.hid_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(self.hid_size, self.out_size)
        self.model = torch.nn.Sequential(self.fc_in, self.relu, self.fc_out)  # 3072x200 
        self.optim = torch.optim.Adam(self.parameters(), lr=0.01)
    
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        out = self.model(x)
        return out

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

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    #print(train_set[0], train_labels[0])
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
