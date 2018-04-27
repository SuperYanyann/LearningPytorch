# -*- coding: UTF-8 -*-
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20,120,kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(120*9, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(self.conv3_drop(self.conv3(x)))
        x = x.view(-1, 120*9)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

# choose different optimization
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#optimizer = optim.SGD(model.parameters(), lr=args.lr)
#optimizer = optim.RMSprop(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(),lr=0.002)

all_train_losses = []
all_test_losses = []
all_train_error = []
all_test_error = []
plot_every = 1000

def train(epoch):
    model.train()
    current_loss = 0;
    correct_error = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        current_loss = current_loss + loss.data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct_error += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == args.log_interval - 1:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            
            all_train_losses.append(current_loss / args.log_interval)
            all_train_error.append(1- float(correct_error)/len(train_loader))
            current_loss = 0
            correct_error = 0

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    all_test_error.append(1 - float(correct)/len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    all_test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))

def changeXlim(arr1,arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    return len1/len2

# get the photo of train loss and test nll_loss
def getLossPhoto():
    changeLen = changeXlim(all_train_losses,all_test_losses)
    ax1 = plt.subplot(211)
    #ax1.title = ("loss")
    ax2 = ax1.twinx()
    ax1.plot(np.arange(len(all_train_losses)),all_train_losses,'b',label = 'train loss')
    ax2.plot(changeLen*np.arange(len(all_test_losses)),all_test_losses,'r',label = 'test loss')
    handles1,labels1 = ax1.get_legend_handles_labels()
    handles2,labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1[::-1],labels1[::-1],bbox_to_anchor = (0.9,0.8))
    ax2.legend(handles2[::-1],labels2[::-1],bbox_to_anchor = (0.9,0.65))

def getErrorPhoto():
    changeLen = changeXlim(all_train_error,all_test_losses)
    bx1 = plt.subplot(212)
    #bx1.title = ("error")
    bx2 = bx1.twinx()
    bx1.plot(np.arange(len(all_train_error)),all_train_error,'b',label = 'train error')
    bx2.plot(changeLen*np.arange(len(all_test_error)),all_test_error,'r',label = 'test error')
    handles1,labels1 = bx1.get_legend_handles_labels()
    handles2,labels2 = bx2.get_legend_handles_labels()
    bx1.legend(handles1[::-1],labels1[::-1],bbox_to_anchor = (0.9,0.8))
    bx2.legend(handles2[::-1],labels2[::-1],bbox_to_anchor = (0.9,0.65))

# use to change the learning rate
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
getLossPhoto()
getErrorPhoto()
fileName = 'channel_1_10_Adam_3.png'
plt.savefig(fileName)
plt.show()
