#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import OrderedDict
import numpy as np

                
def init_weights(m, gain=1, bias=0, method='kaiming'):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, bias)
        if method == 'kaiming':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif method == 'ortho':
            nn.init.orthogonal_(m.weight, gain)
        elif method == 'xavier':
            nn.init.xavier_uniform_(m.weight, gain)
        else:
            raise ValueError()
    elif isinstance(m, (nn.LSTM, nn.RNN, nn.GRU)):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            else:
                raise ValueError()
                
                

class ActorCritic(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM, RNN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        
        # actor
        self.rnn1 = RNN(input_size=STATE_DIM, hidden_size=RNN_SIZE)
        # actor button
        self.l1 = nn.Linear(RNN_SIZE, self.ACTION_DIM)
        # critic
        self.rnn2 = RNN(input_size=STATE_DIM, hidden_size=RNN_SIZE)
        self.l2 = nn.Linear(RNN_SIZE, 1)
        
        # init networks
        init_weights(self.rnn1); init_weights(self.rnn2) 
        init_weights(self.l1, gain=0.01, method='ortho')
        init_weights(self.l2, gain=1, method='ortho')
        
    
    def construct_dist(self, x):  
        logits = self.l1(x)
        dist = Categorical(logits=logits)
        return dist

    def act(self, x, hidden_in, enable_noise=True, return_dist=False): 
        x, hidden_out = self.rnn1(x, hidden_in)
        dist = self.construct_dist(x)
        if enable_noise:
            action = dist.sample()
        else:
            action = dist.mode
        action_logprob = dist.log_prob(action)
        
        if return_dist:
            return action.unsqueeze(-1), action_logprob.unsqueeze(-1), hidden_out, dist
        else:
            return action.unsqueeze(-1), action_logprob.unsqueeze(-1), hidden_out
        
    def evaluate(self, x, hidden_in=None):
        if hidden_in is None:
            x, hidden_out = self.rnn2(x)
        else:
            x, hidden_out = self.rnn2(x, hidden_in)
        
        v = self.l2(x)
        return v, hidden_out
        
    def forward(self, x, action, hidden_in=None): 
        # actor
        a, _ = self.rnn1(x, hidden_in[0])
        dist = self.construct_dist(a)
        action_logprob = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy()
        # critic
        v, _ = self.rnn2(x, hidden_in[1])
        v = self.l2(v)
       
        return action_logprob.unsqueeze(-1), v, dist_entropy.unsqueeze(-1)
    
