# rl-rs-trader

### Additional resources
  - Pdf report of project details (Final-Project-Report.pdf in repo)
  - Slide show of project overview (Final-Slides in repo)
  - Link to slide presentation - https://youtu.be/LddPK043qfw

### Required packages (should be obtainable from pip-install)
  - torch
  - numpy
  - matplotlib
  - seaborn
  - pandas

### You can run this code and running the rl-mercher.py file with the following parameters
 - <agent_type> either TD or DQN
 - <model_load> names are only required if continuing to train an existing model that needs to be loaded
 - <model_targets> only needed if doing dqn because they require the two model files

### if training. 
python rl-mercher.py train <model_save_name> <agent_type> <model_load_name> <model_target_load_name>

### if evaluating.
 python rl-mercher.py eval <eval_model_name> <agent_type> <model_eval_name> <model_target_eval_name>

# Explanation of files

### rl_mercher.py
 main file to run and train models
 runnable via command line arguments as mentioned above
 other paramters configurable by editing the file iteself (random-decay, minimum_randomness)

### looper_helpers.py
 various helper methods using in rl_mercher to clean up main file
 includes the functions handling the learning episode loop and the model evaluation loop

### ge_env.py
 Open-Gym enviornment to simulate trading items on the grand exchange


### td_agent.py
 Temporal Difference Agent 

### dqn_agent.py
 Deep Q-Learning Network Agent

### item_data.csv
 Data file with the historic price/volume information of different game items
