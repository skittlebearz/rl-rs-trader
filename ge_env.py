import gym
from gym import spaces
import numpy as np
import random

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 260
INITIAL_ACCOUNT_BALANCE = 1000000 #1mil

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, maxSteps=27000):
        super(StockTradingEnv, self).__init__()
        self.max_steps = maxSteps

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.window_size = window_size

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=0, high=3, shape=(1,1), dtype=np.int8)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 1), dtype=np.float16)

    def _next_observation(self):
        frame = np.array([
            (self.df.loc[self.current_step: self.current_step+self.window_size, 'avgHighPrice'].values + self.df.loc[self.current_step: self.current_step+self.window_size, 'avgLowPrice'].values) / (MAX_SHARE_PRICE * 2),
            (self.df.loc[self.current_step: self.current_step+self.window_size, 'highPriceVolume'].values / MAX_SHARE_PRICE),# + self.df.loc[self.current_step: self.current_step+5, 'lowPriceVolume'].values) / MAX_SHARE_PRICE,
            (self.df.loc[self.current_step: self.current_step+self.window_size, 'lowPriceVolume'].values / MAX_SHARE_PRICE)
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            self.balance / MAX_ACCOUNT_BALANCE,
            self.items_held / MAX_NUM_SHARES,
        ])

        #print(obs)
        #print(self.balance)
        #print(self.items_held)

        if(np.isnan(obs)).any():
          if self.current_step > len(self.df) or self.current_step > self.max_steps:
            return None
          self.current_step += 1
          return self._next_observation()

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = (self.df.loc[self.current_step, "avgHighPrice"] + self.df.loc[self.current_step, "avgLowPrice"]) / 2

        action_type = action

        if action_type == 1:
            # Buy amount of item
            total_possible = int(self.balance / current_price)
            total_possible = 1000 if total_possible >= 1000 else total_possible 
            shares_bought = int(total_possible)
            prev_cost = self.cost_basis * self.items_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.items_held + shares_bought)
            self.items_held += shares_bought

        elif action_type == 2:
            shares_sold = 1000 if self.items_held >= 1000 else self.items_held 
            self.balance += shares_sold * current_price * .99
            self.items_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.items_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.items_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'avgHighPrice'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / self.max_steps)
        #delay_modifier = .99 - .00004 * self.current_step
        # delay_modifier = .99 - .00002 * self.current_step
        # if self.total_shares_sold < 0:

        # reward = self.net_worth - 10 * self.current_step
        #reward = self.balance - 10 * self.current_step
        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 100000 or self.current_step >= self.max_steps

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.items_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        #self.current_step = random.randint(0, len(self.df.loc[:, 'avgHighPrice'].values) - 6)
        self.current_step = random.randint(0, 7000)

        return self._next_observation()
 
    def resetForEnd(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.items_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        #self.current_step = random.randint(0, len(self.df.loc[:, 'avgHighPrice'].values) - 6)
        # self.current_step = 27005
        self.current_step = 1000

        return self._next_observation()
    
      
    def getNetWorth(self):
      return self.net_worth

    def getProfit(self):
      return self.net_worth - INITIAL_ACCOUNT_BALANCE

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Items held: {self.items_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held items: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

