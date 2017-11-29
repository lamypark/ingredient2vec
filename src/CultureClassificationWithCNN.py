#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import Config
from utils import DataLoader, GensimModels, DataPlotter

# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = Config.path_culture

# Model Hyperparameters
feat_dim = 50
cnn_w = [1,2,3,5]
cnn_h = [25,50,75,125]
sum_h = sum(cnn_h)

# Training Parameters
batch_size = 20
num_epochs = 30
learning_rate = 0.001
momentum = (0.9, 0.999)
evaluate_every = 3

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
dl = DataLoader.DataLoader()
id2cult, id2comp, train_cult, train_comp, train_comp_len, test_cult, test_comp, test_comp_len, max_comp_cnt = dl.load_data(train_file)

print("Train/Test/Cult/Comp: {:d}/{:d}/{:d}/{:d}".format(len(train_cult), len(test_cult), len(id2cult), len(id2comp)))
print("==================================================================================")

class ConvModule(nn.Module):
    def __init__(self, input_size, kernel_sizes, comp_cnt):
        super(ConvModule, self).__init__()

        # attributes:
        self.maxlen = max_comp_cnt
        self.in_channels = input_size
        self.out_channels = len(id2cult) #[25*k for k in kernel_sizes]
        self.cnn_kernel_size = 3 #kernel_sizes
        self.mp_kernel_size = self.maxlen - self.cnn_kernel_size + 1

        # modules:
        self.comp_weight = nn.Embedding(comp_cnt, feat_dim).type(ftype)
        self.cnn1 = nn.Conv1d(self.in_channels, 32, self.cnn_kernel_size)
        self.cnn2 = nn.Conv1d(32, self.out_channels, self.cnn_kernel_size)
        '''
        self.cnn = nn.ModuleList([nn.Conv1d(self.in_channels, self.out_channels[i], 
                                kernel_size=self.cnn_kernel_size[i], stride=1)
                                for i in xrange(len(kernel_sizes))])
        '''
        self.maxpool1 = nn.MaxPool1d(3)
        self.maxpool2 = nn.MaxPool1d(18)
        self.relu = nn.ReLU()

    def forward(self, composer, emb_mask):
        composer = self.comp_weight(composer)
        composer = torch.mul(composer, emb_mask).permute(0,2,1)

        output = self.cnn1(composer)
        output = torch.squeeze(self.maxpool1(output))
        output = self.relu(output)

        output = self.cnn2(output)
        output = torch.squeeze(self.maxpool2(output))
        output = self.relu(output)
        '''
        output = []
        for i, conv in enumerate(self.cnn):
            output.append(torch.max(F.tanh(conv(composer)), dim=2)[0])
        output = torch.cat(output, 1)
        '''

        return output

class LinearModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModule, self).__init__()

        # attributes:
        self.input_size = input_size #275
        self.hidden_size1 = 128
        self.hidden_size2 = 64
        self.output_size = output_size

        # modules:
        #self.linear1 = nn.Linear(self.input_size, self.hidden_size1)
        self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.active = nn.Sigmoid()

    def forward(self, x):
        #x = self.active(self.linear1(x))
        x = self.active(self.linear2(x))

        return x

def parameters():
    params = []
    for model in [cnn_model, linear_model]:
        params += list(model.parameters())

    return params

def make_mask(maxlen, dim, length):
    one = [1]*dim
    zero = [0]*dim
    mask = []
    for c in length:
        mask.append(one*c + zero*(maxlen-c))

    # (batch) * maxlen * dim 
    # [[1 1 1 ... 1 0 0 0 ... 0]...]
    return Variable(torch.from_numpy(np.asarray(mask)).type(ftype), requires_grad=False)

def run(culture, composer, composer_cnt, step):

    optimizer.zero_grad()

    # (batch)
    culture = Variable(torch.from_numpy(np.asarray(culture))).type(ltype)
    # (batch) x (max_comp_cnt(65))
    composer = Variable(torch.from_numpy(np.asarray(composer))).type(ltype)
    emb_mask = make_mask(max_comp_cnt, feat_dim, composer_cnt).view(-1, max_comp_cnt, feat_dim)
    # (batch) x (sum(25 * cnn_w)(275))
    cnn_output = cnn_model(composer, emb_mask)
    # (batch) x (culture_cnt)
    #lin_output = linear_model(cnn_output)

    J = loss_model(cnn_output, culture) 

    cnn_output = np.argmax(cnn_output.data.cpu().numpy(), axis=1)
    culture = culture.data.cpu().numpy()
    hit_cnt = np.sum(np.array(cnn_output) == np.array(culture))

    if step == 2:
        return hit_cnt, J.data.cpu().numpy()
    
    J.backward()
    optimizer.step()
    
    return hit_cnt, J.data.cpu().numpy()

def print_score(batches, step):
    total_hc = 0.0
    total_loss = 0.0

    for i, batch in enumerate(batches):
        batch_cult, batch_comp, batch_comp_len = zip(*batch)
        batch_hc, batch_loss = run(batch_cult, batch_comp, batch_comp_len, step=step)
        total_hc += batch_hc
        total_loss += batch_loss

    print("loss: ", total_loss/i)
    print("acc.: ", total_hc/len(test_cult)*100)
    if step == 3:
        np.save("composer_weight.npy", cnn_model.comp_weight.weight.data.cpu().numpy())
        np.save("id2comp.npy", id2comp)

###############################################################################################
cnn_model = ConvModule(feat_dim, cnn_w, len(id2comp)).cuda()
linear_model = LinearModule(sum_h, len(id2cult)).cuda()
loss_model = nn.CrossEntropyLoss().cuda()
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(parameters(), lr=learning_rate, betas=momentum)

for i in xrange(num_epochs):
    # Training
    train_batches = dl.batch_iter(list(zip(train_cult, train_comp, train_comp_len)), batch_size)
    total_hc = 0.
    total_loss = 0.
    for j, train_batch in enumerate(train_batches):
        batch_cult, batch_comp, batch_comp_len = zip(*train_batch)
        batch_hc, batch_loss = run(batch_cult, batch_comp, batch_comp_len, step=1)
        total_hc += batch_hc
        total_loss += batch_loss
        if (j+1) % 500 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, "acc. :", total_hc/batch_size/j*100, datetime.datetime.now()

    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1))
        test_batches = data_loader.batch_iter(list(zip(test_cult, test_comp, test_comp_len)), batch_size)
        print_score(test_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = data_loader.batch_iter(list(zip(test_cult, test_comp, test_comp_len)), batch_size)
print_score(test_batches, step=3)
