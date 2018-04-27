import argparse
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=64,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=128,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=3,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
from data_loader import DataLoader

data = torch.load(args.data)
args.max_len = data["max_len"]
args.vocab_size = data['dict']['vocab_size']
args.label_size = data['dict']['label_size']

training_data = DataLoader(
             data['train']['src'],
             data['train']['label'],
             args.max_len,
             batch_size=args.batch_size,
             cuda=use_cuda)

validation_data = DataLoader(
              data['valid']['src'],
              data['valid']['label'],
              args.max_len,
              batch_size=args.batch_size,
              shuffle=False,
              cuda=use_cuda)

              
# ##############################################################################
# Build model
# ##############################################################################
import model

rnn = model.LSTM_Text(args)
if use_cuda:
    rnn = rnn.cuda()

# optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=0.001)
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
train_error = []
valid_error = []
accuracy = []

def repackage_hidden(h):
    if type(h) == Variable:
        if use_cuda:
            return Variable(h.data).cuda()
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate():
    rnn.eval()
    corrects = eval_loss = 0
    _size = validation_data.sents_size
    hidden = rnn.init_hidden()
    for data, label in tqdm(validation_data, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        hidden = repackage_hidden(hidden)
        pred, hidden = rnn(data, hidden)
        loss = criterion(pred, label)

        eval_loss += loss.data
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

    return eval_loss[0]/_size, corrects, corrects/_size * 100.0, _size

def train():
    rnn.train()
    total_loss = 0
    corrects  = 0
    hidden = rnn.init_hidden()
    for data, label in tqdm(training_data, mininterval=1,
                desc='Train Processing', leave=False):
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        target, hidden = rnn(data, hidden)
        loss = criterion(target, label)
        corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss[0]/training_data.sents_size,corrects/training_data.sents_size * 100.0

def changeXlim(arr1,arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    return len1/len2

# get the photo of train loss and test nll_loss
def getLossPhoto():
    changeLen = changeXlim(train_loss,valid_loss)
    ax1 = plt.subplot(211)
    #ax1.title = ("loss")
    ax2 = ax1.twinx()
    ax1.plot(np.arange(len(train_loss)),train_loss,'b',label = 'train loss')
    ax2.plot(changeLen*np.arange(len(valid_loss)),valid_loss,'r',label = 'test loss')
    handles1,labels1 = ax1.get_legend_handles_labels()
    handles2,labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1[::-1],labels1[::-1],bbox_to_anchor = (0.9,0.8))
    ax2.legend(handles2[::-1],labels2[::-1],bbox_to_anchor = (0.9,0.65))

def getErrorPhoto():
    changeLen = changeXlim(train_error,valid_error)
    bx1 = plt.subplot(212)
    #bx1.title = ("error")
    bx2 = bx1.twinx()
    bx1.plot(np.arange(len(train_error)),train_error,'b',label = 'train error')
    bx2.plot(changeLen*np.arange(len(valid_error)),valid_error,'r',label = 'test error')
    handles1,labels1 = bx1.get_legend_handles_labels()
    handles2,labels2 = bx2.get_legend_handles_labels()
    bx1.legend(handles1[::-1],labels1[::-1],bbox_to_anchor = (0.9,0.8))
    bx2.legend(handles2[::-1],labels2[::-1],bbox_to_anchor = (0.9,0.65))

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss,train_acc = train()
        train_loss.append(loss*1000.)
        train_error.append(100- train_acc)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        valid_error.append(100-acc)
        accuracy.append(acc)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = rnn.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
                "src_dict": data['dict']['train']
            }
            torch.save(model_source, args.save)

    # get photo and save
    getLossPhoto()
    getErrorPhoto()
    fileName = 'test.png'
    plt.savefig(fileName)
    plt.show()

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
