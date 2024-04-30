import datetime
import torch
from torch.nn import LSTM
import numpy as np
from torch.optim import Adam
from pathlib import Path


class ConfigCore():
    def __init__(self, data_path='./'):
        # network
        self.device = 'cuda:0'
        self.SEED_NUMBER = 0
        self.FC_SIZE = 256
        self.RNNSELF = LSTM
        self.RNNSELF_SIZE = 256
        
        # RL
        self.GAMMA = 0.9
        self.num_epoch = 80
        self.LAMBDA = 1  # for GAE
        self.eps_clip = 0.2
        self.entropy_coef = torch.tensor([0.01])
        
        # optimzer
        self.optimzer = Adam
        self.lr = 3e-4
        self.eps = 1e-3
        self.decayed_lr = 5e-5
        self.batch_size = 500
        
        # environment
        self.STATE_DIM = 4 + 1 + 1 + 1   # obs, last action and last reward and t
        self.ACTION_DIM = 4
        self.REWARD_SCALE = 10
        self.truncated_len = 1000
        self.full_len = 1000
        
        # others
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = data_path
        
        
    def save(self):
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, self.data_path / f'{self.filename}_arg.pkl')
        
    def load(self, filename):
        self.__dict__ = torch.load(self.data_path / f'{filename}_arg.pkl')
        self.filename = filename
        