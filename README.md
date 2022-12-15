# rl-rs-trader

### You can run this code and running the rl-mercher.py file with the following parameters
 - <agent_type> either TD or DQN
 - <model_load> names are only required if continuing to train an existing model that needs to be loaded
 - <model_targets> only needed if doing dqn because they require the two model files

### if training. 
python rl-mercher.py train <model_save_name> <agent_type> <model_load_name> <model_target_load_name>

### if evaluating.
 python rl-mercher.py eval <eval_model_name> <agent_type> <model_eval_name> <model_target_eval_name>
