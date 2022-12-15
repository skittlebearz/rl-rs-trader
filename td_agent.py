import numpy as np
import torch

#@title PolicyNN
class PolicyNN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNN, self).__init__()
        self.layer1 = torch.nn.Linear(observation_space, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 32)
        self.layer4 = torch.nn.Linear(32, action_space)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer4(x)

        return x


#@title Agent
class TDAgent(object):
    def __init__(self, decay, min_randomness):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNN(32, 3).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)
        self.decay = decay
        self.randomness = 1.0
        self.min_randomness = min_randomness

    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.from_numpy(state).float().to(self.device)

        # find the quality of both actions
        qualities = self.model(state).cpu()

        # sometimes take a random action
        if np.random.rand() <= self.randomness:
            action = np.random.randint(low=0, high=3)
        else:
            action = torch.argmax(qualities).item()

        # return that action
        return action

    def update(self, state, next_state, reward):
      st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
      next_st = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
      old_target = self.model(st)
      new_target = reward + torch.amax(self.model(next_st), dim=0, keepdim=True)
      loss = torch.nn.functional.smooth_l1_loss(old_target, new_target) 
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()


    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)

