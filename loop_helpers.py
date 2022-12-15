import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#given agent and environment, do learning session with the agent through an episode of the env dataset
def learningEpisode(agent_type, lagent, env, cumulative, learning, bestSoFar, bestSoFarModel):
  max_iteration = 20000
  
  state = env.reset()
  
  profit = []
  cumulative_rewards = 0
  
  for iteration in range(1, max_iteration + 1):

      action = lagent.act(state)
      next_state, reward, done, *_ = env.step(action)
      if agent_type.upper() == "DQN":
        lagent.memory.push(state,action,next_state,reward)
      lagent.update(state, next_state, reward)
      state = next_state

      cumulative_rewards += reward 
      learning.append(reward)
      profit.append(env.getProfit())

      if done:
        break

  lagent.update_randomness()
  cumulative.append(cumulative_rewards)
  if cumulative_rewards > bestSoFar:
    if agent_type.upper() == "TD":
      bestSoFarModel = lagent.model.state_dict()
      bestSoFar = cumulative_rewards
    elif agent_type.upper() == "DQN":
      bestSoFarModel = lagent.policy_net.state_dict(), lagent.target_net.state_dict()
      bestSoFar = cumulative_rewards
    

  env.render()
  return(env.getProfit()/len(profit)), bestSoFar, bestSoFarModel




#will do an evaluation of an agent given env and agent ready for evaluations (i.e. - randomness set to 0)
def evalLoopEnd(lagent, env):
  max_iteration = 10000
  state = env.resetForEnd()
  profit = []
  
  #loop of agent acting and keeping track of profit through the trades
  for iteration in range(1, max_iteration + 1):
      action = lagent.act(state)
      next_state, reward, done, *_ = env.step(action)
      profit.append(env.getProfit())
      state = next_state
      if done:
        break

  #show the graph of profit of agent on the unseen data
  x = np.arange(0, len(profit), 1)
  y = profit
  sns.lineplot(x=x, y=y)
  plt.title("Profit of Agent on Unseen Data")
  plt.xlabel("Step")
  plt.ylabel("Profit")
  plt.show()
 
  env.render()
  return(env.getProfit()/len(profit))

