#! /usr/bin/env python

import os
import datetime
import numpy as np
import random
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
id2cult, id2ingr, train_cult, train_ingr, train_ingr_len, test_cult, test_ingr, test_ingr_len, max_ingr_cnt, ingrid2vec = dl.load_data(train_file, feat_dim)

print("Train/Test/Cult/ingr: {:d}/{:d}/{:d}/{:d}".format(len(train_cult), len(test_cult), len(id2cult), len(id2ingr)))
print("==================================================================================")

class CBOWModule(nn.Module):
    def __init__(self, input_size, ingr_cnt):
        super(CBOWModule, self).__init__()

        # attributes:
        self.input_size = input_size
        self.hidden_size1 = 256

        # modules:
        self.ingr_weight = nn.Embedding(ingr_cnt, feat_dim).type(ftype)
        self.ingr_weight.weight.data.copy_(torch.from_numpy(np.asarray(ingrid2vec)))
        self.linear1 = nn.Linear(self.input_size, self.hidden_size1)
        self.linear2 = nn.Linear(self.hidden_size1, len(id2ingr))
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)

    def forward(self, x, emb_mask, x_cnt, step):
        b_s = x.shape[0]
        idx = []
        for i in xrange(b_s):
            idx.append(random.sample([j for j in xrange(x_cnt[i])], x_cnt[i])\
                        +[j for j in xrange(x_cnt[i], max_ingr_cnt)])
        target = np.asarray([x[i][l[0]].data.cpu().numpy() for i, l in enumerate(idx)])
        target = Variable(torch.from_numpy(np.asarray(target)).view(-1)).type(ltype)
        idx = [a+i*max_ingr_cnt for i, b in enumerate(idx) for a in b[1:]]
        x = self.ingr_weight(x)
        x = torch.mul(x, emb_mask).view(-1, feat_dim)
        x = x[idx].view(-1, max_ingr_cnt-1, feat_dim)
        x = torch.sum(x, 1)
        x = torch.cat([(x[i]/(x_cnt[i]-1)).view(1,feat_dim) for i in xrange(b_s)], 0)
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return out, target

def parameters():
    params = []
    for model in [cbow_model]:
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

def run(ingredient, ingredient_cnt, step):

    optimizer.zero_grad()

    # (batch) x (max_ingr_cnt(65))
    ingredient = Variable(torch.from_numpy(np.asarray(ingredient))).type(ltype)
    emb_mask = make_mask(max_ingr_cnt, feat_dim, ingredient_cnt).view(-1, max_ingr_cnt, feat_dim)
    cbow_output, target = cbow_model(ingredient, emb_mask, ingredient_cnt, step)

    J = loss_model(cbow_output, target) 

    if step > 1:
        cbow_output = np.argsort(-1*cbow_output.data.cpu().numpy(), axis=1)
        target = target.data.cpu().numpy()
        return cbow_output, target 
    
    J.backward()
    optimizer.step()
    
    return J.data.cpu().numpy()

def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.

    for batch in batches:
        batch_ingr, batch_ingr_len = zip(*batch)
        batch_o, target = run(batch_ingr, batch_ingr_len, step=step)

        recall1 += np.sum([target[i] in batch_o[i][:1] for i in xrange(len(target))]) 
        recall5 += np.sum([target[i] in batch_o[i][:5] for i in xrange(len(target))]) 
        recall10 += np.sum([target[i] in batch_o[i][:10] for i in xrange(len(target))]) 
        recall100 += np.sum([target[i] in batch_o[i][:100] for i in xrange(len(target))]) 
        recall1000 += np.sum([target[i] in batch_o[i][:1000] for i in xrange(len(target))]) 

    test_len = len(test_cult)
    print("recall@1: ", recall1/test_len)
    print("recall@5: ", recall5/test_len)
    print("recall@10: ", recall10/test_len)
    print("recall@100: ", recall100/test_len)
    print("recall@1000: ", recall1000/test_len)

    '''
    if step == 3:
        np.save("ingredient_weight.npy", linear_model.ingr_weight.weight.data.cpu().numpy())
        np.save("id2ingr.npy", id2ingr)
    '''

###############################################################################################
cbow_model = CBOWModule(feat_dim, len(id2ingr)).cuda()
loss_model = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(parameters(), lr=learning_rate, betas=momentum)
#optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)

for i in xrange(num_epochs):
    # Training
    train_batches = dl.batch_iter(list(zip(train_ingr, train_ingr_len)), batch_size)
    total_loss = 0.
    for j, train_batch in enumerate(train_batches):
        batch_ingr, batch_ingr_len = zip(*train_batch)
        batch_loss = run(batch_ingr, batch_ingr_len, step=1)
        total_loss += batch_loss
        if (j+1) % 1000 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()

    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1))
        test_batches = dl.batch_iter(list(zip(test_ingr, test_ingr_len)), batch_size)
        print_score(test_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = dl.batch_iter(list(zip(test_ingr, test_ingr_len)), batch_size)
print_score(test_batches, step=3)
