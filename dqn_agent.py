import torch
import random
from collections import namedtuple
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.main = torch.nn.Sequential(
			torch.nn.Linear(state_size, 64),
			torch.nn.LeakyReLU(0.01, inplace=True),
			torch.nn.Linear(64, 64),
			torch.nn.LeakyReLU(0.01, inplace=True),
			torch.nn.Linear(64, 32),
			torch.nn.LeakyReLU(0.01, inplace=True),
			torch.nn.Linear(32, action_size),
			# torch.nn.Linear(32, action_size),
        # x = torch.nn.functional.softmax(x, dim=1)
		)
	
	def forward(self, input):
		return self.main(input)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
	def __init__(self, decay, min_randomness, is_eval=False):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.state_size = 32 # normalized previous days
		self.action_size = 3 # sit, buy, sell
		# self.memory = ReplayMemory(10000)
		self.memory = ReplayMemory(800)
		# self.inventory = []
		self.is_eval = is_eval
		# self.i=0

		self.gamma = 0.95
		self.randomness = 1.0
		self.epsilon_min = min_randomness
		self.epsilon_decay = decay
		self.batch_size = 256 #32!!!!!!!!!!!!!!
		self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
		self.target_net = DQN(self.state_size, self.action_size).to(self.device)
		self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.005, momentum=0.9)
		# self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=4e-4, momentum=0.9)

	def act(self, state):
		#if not self.is_eval and np.random.rand() <= self.randomness:
		if np.random.rand() <= self.randomness:
		# if self.i < 5:
			# self.i+=1
			# print("hi")
			return random.randrange(self.action_size)

		tensor = torch.FloatTensor(state).to(device)
		options = self.target_net(tensor)
		return np.argmax(options[0].detach().numpy())

	#def optimize(self):
	def update(self, st, nst, fini):
		if len(self.memory) < self.batch_size:
				return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(batch.next_state).to(device)
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.stack([s for s in next_state if s is not None])
		# non_final_next_states = torch.cat(next_state)
		state_batch = torch.FloatTensor(batch.state).to(device)
		action_batch = torch.LongTensor(batch.action).to(device)
		reward_batch = torch.FloatTensor(batch.reward).to(device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		state_action_values = self.policy_net(state_batch).reshape((self.batch_size, 3)).gather(1, action_batch.reshape((self.batch_size, 1)))

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		next_state_values = torch.zeros(self.batch_size, device=device)
		#print(self.target_net(non_final_next_states))
		next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		
		# Compute the expected Q values
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		# Compute Huber loss
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
				param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

	def update_randomness(self):
		if self.randomness > self.epsilon_min:
			self.randomness = self.randomness * self.epsilon_decay