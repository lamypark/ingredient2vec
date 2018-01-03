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
feat_dim = 10

# Training Parameters
batch_size = 10
num_epochs = 30
learning_rate = 0.005
momentum = (0.9, 0.99)
evaluate_every = 3

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
dl = DataLoader.DataLoader()
id2cult, id2ingr, train_cult, train_ingr, train_ingr_len, valid_cult, valid_ingr, valid_ingr_len, test_cult, test_ingr, test_ingr_len, max_ingr_cnt, ingrid2vec = dl.load_data(train_file, feat_dim)

print("Train/Valid/Test/Cult/ingr: {:d}/{:d}/{:d}/{:d}/{:d}".format(len(train_cult), len(valid_cult), len(test_cult), len(id2cult), len(id2ingr)))
print("==================================================================================")

class RNNModule(nn.Module):
    def __init__(self, cult_cnt, ingr_cnt):
        super(RNNModule, self).__init__()

        # modules:
        self.ingr_weight = nn.Embedding(ingr_cnt, feat_dim, padding_idx=0).type(ftype)
        self.ingr_weight.weight.data.copy_(torch.from_numpy(np.asarray(ingrid2vec)))
        self.gru = nn.GRU(feat_dim, cult_cnt, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.active = nn.Sigmoid()

    def forward(self, x, emb_mask, x_len, step):
        x = self.ingr_weight(x)
        x = torch.mul(x, emb_mask)
        print x_len
        x_len, x_idx = torch.sort(x_len, descending=True)
        x, _ = self.gru(x)
        x = [x[i,x_len[i]-1,:].view(1,-1) for i in xrange(x.size()[0])]
        x = self.active(torch.cat(x, 0))

        return x

def parameters():
    params = []
    for model in [rnn_model]:
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

def run(culture, ingredient, ingredient_cnt, step):

    optimizer.zero_grad()

    # (batch)
    culture = Variable(torch.from_numpy(np.asarray(culture))).type(ltype)
    # (batch) x (max_ingr_cnt(65))
    ingredient = Variable(torch.from_numpy(np.asarray(ingredient))).type(ltype)
    emb_mask = make_mask(max_ingr_cnt, feat_dim, ingredient_cnt).view(-1, max_ingr_cnt, feat_dim)
    #cnn_output = cnn_model(ingredient, emb_mask)
    rnn_output = rnn_model(ingredient, emb_mask, ingredient_cnt, step)

    J = loss_model(rnn_output, culture) 

    rnn_output = np.argmax(rnn_output.data.cpu().numpy(), axis=1)
    culture = culture.data.cpu().numpy()
    hit_cnt = np.sum(np.array(rnn_output) == np.array(culture))

    if step == 2:
        return hit_cnt, J.data.cpu().numpy()
    
    J.backward()
    optimizer.step()
    
    return hit_cnt, J.data.cpu().numpy()

def print_score(batches, step):
    total_hc = 0.0
    total_loss = 0.0

    rnn_model.eval()
    for i, batch in enumerate(batches):
        batch_cult, batch_ingr, batch_ingr_len = zip(*batch)
        batch_hc, batch_loss = run(batch_cult, batch_ingr, batch_ingr_len, step=step)
        total_hc += batch_hc
        total_loss += batch_loss

    print("loss: ", total_loss/i)
    print("acc.: ", total_hc/len(test_cult)*100)
    '''
    if step == 3:
        np.save("ingredient_weight.npy", rnn_model.ingr_weight.weight.data.cpu().numpy())
        np.save("id2ingr.npy", id2ingr)
    '''

###############################################################################################
rnn_model = RNNModule(len(id2cult), len(id2ingr)).cuda()
loss_model = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(parameters(), lr=learning_rate, betas=momentum)
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)

for i in xrange(num_epochs):
    # Training
    rnn_model.train()
    train_batches = dl.batch_iter(list(zip(train_cult, train_ingr, train_ingr_len)), batch_size)
    total_hc = 0.
    total_loss = 0.
    rnn_model.train()
    for j, train_batch in enumerate(train_batches):
        batch_cult, batch_ingr, batch_ingr_len = zip(*train_batch)
        batch_hc, batch_loss = run(batch_cult, batch_ingr, batch_ingr_len, step=1)
        total_hc += batch_hc
        total_loss += batch_loss
        if (j+1) % 1000 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, "acc. :", total_hc/batch_size/j*100, datetime.datetime.now()

    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1))
        test_batches = dl.batch_iter(list(zip(valid_cult, valid_ingr, valid_ingr_len)), batch_size)
        print_score(test_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = dl.batch_iter(list(zip(test_cult, test_ingr, test_ingr_len)), batch_size)
print_score(test_batches, step=3)
