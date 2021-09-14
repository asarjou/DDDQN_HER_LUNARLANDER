import torch
import random

class E_Greedy_Policy():
    
    def __init__(self, epsilon, decay, min_epsilon):
        #initialise parameters
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.decay = decay
        self.epsilon_min = min_epsilon
        
    def __call__(self, state, n_actions, device, Q_network):
        ###borrowed and adapted from Lab 6###
        is_greedy = random.random() > self.epsilon #if is greedy
        
        if is_greedy :
            # we select greedy action
            with torch.no_grad():
                Q_network.eval() 
                index_action = Q_network(state).max(1)[1].view(1, 1) #take greedy action 
                Q_network.train() 
        else:
            index_action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long) #take random action
        
        return index_action
                
    def update_epsilon(self):
        
        self.epsilon = self.epsilon*self.decay
        if self.epsilon < self.epsilon_min: #can't go below min epsilon
            self.epsilon = self.epsilon_min
        
    def reset(self):
        self.epsilon = self.epsilon_start