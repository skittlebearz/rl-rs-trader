from pickle import GET
import gym
from gym import spaces
import numpy as np
import random


MAX_CASH_STACK = 2147483647 #2.1 bil
INITIAL_ACCOUNT_BALANCE = 1000000 #1 mil

class GETradingEnv(gym.Env):
  """A custom gym interface for trading items on the Grand Exchange in the game Old School Runescape"""
  metadata = {'render.modes': ['human']}

  def __init__(self, item_frame):
    super(GETradingEnv, self).__init__()
    self.item_frame = item_frame
    self.reward_range = (0, MAX_CASH_STACK)
    self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=1, shape=
                    (7, 2) , dtype=np.float16)

  def step(self, action):
    self.balance = INITIAL_ACCOUNT_BALANCE
    self.net_worth = INITIAL_ACCOUNT_BALANCE
    self.max_net_worth = INITIAL_ACCOUNT_BALANCE
    self.items_held = 0
    self.cost_basis = 0
    self.total_items_sold = 0
    self.total_sales_value = 0

    #TODO make this apply
    self.current_step = random.randint(0, len(self.item_frame.loc[:, 'Open'].values) - 6)
    return self._next_observation()

  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...