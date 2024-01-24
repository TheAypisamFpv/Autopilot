# #deep q learning ai, intput sensor: image of the road, sterring angle, gas, break, acceleration

####################
# WORK IN PROGRESS #
####################
 
# import torch as T
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np

# class DeepQNetwork(nn.Module):
#     def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
#         super(DeepQNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.loss = nn.MSeLoss()
#         self.devie = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         actions = self.fc3(x)

#         return actions

# class Agent():
#     def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100_000, eps_end=0.01, eps_dec=5e-4):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.eps_min = eps_end
#         self.eps_dec = eps_dec
#         self.lr = lr
#         self.action_space = [i for in range(n_actions)]
#         self.mem_size = max_mem_size
#         self.batch_size = batch_size
#         self.mem_cntr = 0

#         self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        