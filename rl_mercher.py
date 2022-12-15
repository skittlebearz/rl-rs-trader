#@title tradingenv
import pandas as pd
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from ge_env import StockTradingEnv
from loop_helpers import *
from td_agent import TDAgent
from dqn_agent import DQNAgent


def trainAgent():
    model_load_name = ""
    target_model_load_name = ""
    args = sys.argv
    print(args)
    agent_type = args[1]
    model_save_name = args[2]
    if len(args) > 3:
        model_load_name = args[3]
    if len(args) > 4:
        target_model_load_name = args[4]

    itemDF = pd.read_csv('./item_data.csv', index_col=0)
    itemDF = itemDF[itemDF['id'] == 2]
    itemDF = itemDF.drop('dt_updated',axis=1)

    bestSoFarModel = None
    bestSoFar = 0
    learning = []
    cumulativeReward = []
    cumulativeRewardAverage = []

    agent = None
    environment = None
    if agent_type.upper() == "TD":
        agent = TDAgent(.997, .1)
        environment = StockTradingEnv(itemDF, 9)
    elif agent_type.upper() == "DQN":
        agent = None
        environment = StockTradingEnv(itemDF, 9)
        agent = DQNAgent(.997, 0.1)
    
    if agent_type.upper() == "TD" and model_load_name != "":
        agent.model.load_state_dict(torch.load(f"./{model_load_name}.pth"))

    if agent_type.upper() == "DQN" and model_load_name != "":
        agent.policy_net.load_state_dict(torch.load(f"./{model_load_name}.pth"))
        agent.target_net.load_state_dict(torch.load(f"./{target_model_load_name}.pth"))

    runavgtotal = 0;
    #Train Model
    for i in range(1500):
     if i%20==0:
        cumulativeRewardAverage.append(runavgtotal/20)
        runavgtotal = 0
     print(i)
     prof, bestSoFar, bestSoFarModel = learningEpisode(agent_type, agent, environment, cumulativeReward, learning, bestSoFar, bestSoFarModel)
     runavgtotal += cumulativeReward[-1]
     print(agent.randomness)

    #Save models
    if agent_type.upper() == "TD":
        torch.save(agent.model.state_dict(), f"./{model_save_name}.pth")
        torch.save(bestSoFarModel, f"./best-{model_save_name}.pth")
    if agent_type.upper() == "DQN":
        torch.save(agent.policy_net.state_dict(), f"./{model_save_name}.pth")
        torch.save(agent.target_net.state_dict(), f"./{model_save_name}-targ.pth")
        mod, targ= bestSoFarModel
        torch.save(mod, f"./best-{model_save_name}.pth")
        torch.save(targ, f"./best-{model_save_name}-targ.pth")


    #test onlastpart
    max_steps = 37000
    environment = StockTradingEnv(itemDF, 9, max_steps)

    agent.randomness = 0
    profits = []
    gainz = evalLoopEnd(agent, environment)
    profits.append(gainz)

    x = np.arange(0, len(cumulativeReward), 1)
    sns.lineplot(x=x, y=cumulativeReward)
    plt.title("reward")
    plt.xlabel("Episodes")
    plt.ylabel("profit")
    plt.show()

    x = np.arange(0, len(cumulativeRewardAverage), 1)
    sns.lineplot(x=x, y=cumulativeRewardAverage)
    plt.title("average reward from 20 episodes")
    plt.xlabel("set of 20 Episodes")
    plt.ylabel("profit")
    plt.show()

def evalAgent():
    model_load_name = ""
    target_model_load_name = ""
    args = sys.argv
    print(args)
    agent_type = args[1]
    if len(args) > 2:
        model_load_name = args[2]
    if len(args) > 3:
        target_model_load_name = args[3]

    itemDF = pd.read_csv('./item_data.csv', index_col=0)
    itemDF = itemDF[itemDF['id'] == 2]
    itemDF = itemDF.drop('dt_updated',axis=1)


    agent = None
    environment = None
    if agent_type.upper() == "TD":
        agent = TDAgent(.997, .1)
        environment = StockTradingEnv(itemDF, 9)
    elif agent_type.upper() == "DQN":
        agent = None
        environment = StockTradingEnv(itemDF, 15)
        agent = DQNAgent(0.997, 0.1, True)
    
    if agent_type.upper() == "TD" and model_load_name != "":
        agent.model.load_state_dict(torch.load(f"./{model_load_name}.pth"))

    if agent_type.upper() == "DQN" and model_load_name != "":
        agent.policy_net.load_state_dict(torch.load(f"./{model_load_name}.pth"))
        agent.target_net.load_state_dict(torch.load(f"./{target_model_load_name}.pth"))


    #test onlastpart
    max_steps = 37000
    environment = StockTradingEnv(itemDF, 9, max_steps)

    agent.randomness = 0
    profits = []
    gainz = evalLoopEnd(agent, environment)
    profits.append(gainz)



def main():
    # trainAgent()
    evalAgent()

if __name__ == "__main__":
    main()