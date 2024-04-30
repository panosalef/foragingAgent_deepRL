#!/usr/bin/env python
# coding: utf-8

# In[25]:
import torch
import torch.nn as nn
import torch.nn.functional as F


class Buffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.actionlogprobs = []
        self.rewards = []
        self.actorhidden0s = []
        self.critichidden0s = 0
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.actionlogprobs.clear()
        self.rewards.clear()
        self.actorhidden0s.clear()
    
    
class Agent():
    def __init__(self, arg, ActorCritic):
        self.__dict__ .update(arg.__dict__)
        self.entropy_coef = self.entropy_coef.to(self.device)
        self.model = ActorCritic(self.STATE_DIM, self.ACTION_DIM,
                                 self.RNNSELF_SIZE, self.FC_SIZE, self.RNNSELF).to(self.device)
        self.model_optim = self.optimzer(self.model.parameters(), lr=self.lr, eps=self.eps)
        self.model.eval()
        self.buffer = Buffer()

    def select_action(self, state, hidden_in, enable_noise=True, return_dist=False):            
        with torch.no_grad():
            action, action_logprob, hidden_out, dist = self.model.act(state, hidden_in, 
                                                                      enable_noise=enable_noise, 
                                                                      return_dist=return_dist)
        return action, action_logprob, hidden_out, dist
    
    def GAE(self, rewards, dones, values_current, values_next):        
        advantages = torch.zeros((rewards.shape[0] + 1, rewards.shape[1], 1), device=self.device)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[[t]] + ((1-dones[[t]]) * self.GAMMA * values_next[[t]]) - values_current[[t]]
            advantages[[t]] = delta + (self.GAMMA * self.LAMBDA * advantages[[t + 1]])
           
        advantages = advantages[:-1]
        Qvalues = advantages + values_current
        return advantages, Qvalues   
    
    def update_parameters(self):    
        states = torch.cat(self.buffer.states, dim=1)
        actions_old = torch.cat(self.buffer.actions, dim=1)
        actionlogprobs_old = torch.cat(self.buffer.actionlogprobs, dim=1)
        rewards = torch.cat(self.buffer.rewards, dim=1)
        dones = torch.zeros_like(rewards)
        dones[-1] = 1
        actorhidden0s = self.buffer.actorhidden0s[0]
        critichidden0s = actorhidden0s
        if actorhidden0s[0].sum() == 0: # hidden state
            critichidden0s = actorhidden0s
        else:
            critichidden0s = self.buffer.critichidden0s
        
        with torch.no_grad():
            values_current_old, critichidden = self.model.evaluate(states, hidden_in=critichidden0s)
        self.buffer.critichidden0s = critichidden
        values_next_old = torch.zeros_like(values_current_old)
        values_next_old[:-1] = values_current_old[1:]
        advantages, values_target = self.GAE(rewards, dones, values_current_old, values_next_old)
        
        for _ in range(self.num_epoch):                
            actionlogprobs, values_current, dist_entropy = self.model(states, actions_old, hidden_in=[actorhidden0s, critichidden0s])
         
            policy_ratios = torch.exp(actionlogprobs - actionlogprobs_old)
            surr1 = policy_ratios * advantages
            surr2 = torch.clamp(policy_ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_current, values_target)
            entropy_bonus = (dist_entropy * self.entropy_coef).mean()
            
            loss = - policy_loss + value_loss - entropy_bonus
            
            self.model_optim.zero_grad()
            loss.backward()
            self.model_optim.step()
            
        #torch.cuda.empty_cache()   
        return policy_loss.detach(), value_loss.detach(), dist_entropy.detach().mean(dim=(0, 1))

    def learn(self):
        self.model.train()
        losses = self.update_parameters()
        self.model.eval()
        return losses

    def save(self, episode, full_param=True):
        file = self.data_path / f'{self.filename}-{episode}.pth.tar'
            
        state = {'model_dict': self.model.state_dict()}
        if full_param:
            state.update({'model_optimizer_dict': self.model_optim.state_dict()})
            
        torch.save(state, file)

    def load(self, filename, load_optimzer):
        self.filename = filename
        file = self.data_path / f'{self.filename}.pth.tar'
        state = torch.load(file)

        self.model.load_state_dict(state['model_dict'])
        
        if load_optimzer is True:
            self.model_optim.load_state_dict(state['model_optimizer_dict'])
            
