import torch
import numpy as np
from collections import namedtuple
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as Tb
import matplotlib.pyplot as plt

### Borrowed and Adapted from Lab 6/Pytorch DQN Tutorial ###
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward')) #Creates a Named tuple for transitions


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity #setting Memory capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, her_arr, goal, device):

        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        state_tensor = Utils.convert_state(state, goal, device)  #concatenates goal to state 
        
        if next_state is None:
            state_tensor_next = None            
        else:
            state_tensor_next = Utils.convert_state(next_state, goal, device) #concatenates goal to next state
            
        action_tensor = torch.tensor([action], device=device).unsqueeze(0)

        reward = torch.tensor([reward], device=device).unsqueeze(0)/10. # 10. # reward scaling

        self.memory[self.position] = Transition(state_tensor, action_tensor, state_tensor_next, reward) #this adds the experieence to the replay buffer
        her_arr.append(Transition(state_tensor, action_tensor, state_tensor_next, reward)) #append the same experience to the her_replay_tracker for HER

        self.position = (self.position + 1) % self.capacity

    def HER_push(self, state, action, next_state, reward, goal, device): #create a push specially for pushing HER experiences

        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        state_tensor = convert_state(state, goal, device)
        
        if next_state is None:
            state_tensor_next = None            
        else:
            state_tensor_next = convert_state(next_state, goal, device)
            
        action_tensor = torch.tensor([action], device=device).unsqueeze(0)

        reward = torch.tensor([reward], device=device).unsqueeze(0)/10. #1000 reward scaling

        self.memory[self.position] = Transition(state_tensor, action_tensor, state_tensor_next, reward) #this adds the experieence to the replay buffer

        self.position = (self.position + 1) % self.capacity
   

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)#random sample based on batch size 

    def __len__(self):
        return len(self.memory)
        
class Utils:
    
    def convert_state(state, goal, device): 
        state_goal = np.concatenate((state, goal)) #concatenate goal to state
        state_tensor = torch.tensor(state_goal, device=device).unsqueeze(0) #convert to tensor
        return state_tensor
    
    def plot_experiments(rewards_history, ema_coefficient = 0.9, warmup = 50): 
        #borrowed and adapted from lab 7
        list_of_experiments = {'test': rewards_history}
        for exp_name, list_total_rewards in list_of_experiments.items():
            
            ema = 0
            list_ema = [] 
            
            for ind, rew in enumerate(list_total_rewards):
                
                ema = ema * ema_coefficient + rew * (1 - ema_coefficient)
                if ind > warmup: 
                    list_ema.append(ema)
            
            plt.plot(list_ema, '-', label = exp_name)
            plt.xlabel('Episodes')
            plt.ylabel('Exponential Moving Average Success')
        
    def optimize_model(device, memory, BATCH_SIZE, Q_network, Q_target, GAMMA,optimizer):
    #borrowed and adapted from lab 6/ pytorch DQN implementation
        transitions = memory.sample(BATCH_SIZE) #take mini batch from replay buffer
        batch = Transition(*zip(*transitions)) #unzip the transitions 
    
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]) 
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q values using policy net
        Q_values = Q_network(state_batch).gather(1, action_batch) #changed dimension to test
    
        # Compute next Q values using Q_targets
        next_Q_values = torch.zeros( BATCH_SIZE, device=device)
    
        selected_actions = Q_target(non_final_next_states).max(1)[1].unsqueeze(1).detach() #select action from target net for DDQN
        next_Q_values[non_final_mask] = Q_network(non_final_next_states).gather(1, selected_actions).view(-1,).detach() #implementing DDQN
        
        next_Q_values = next_Q_values.unsqueeze(1)
        
        # Compute targets
        target_Q_values = (next_Q_values * GAMMA) + reward_batch
        
        # Compute MSE Loss
        loss = F.mse_loss(Q_values, target_Q_values)
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        
        # Trick: gradient clipping
        for param in Q_network.parameters():
            param.grad.data.clamp_(-1, 1)
            
        optimizer.step()
        
        return loss