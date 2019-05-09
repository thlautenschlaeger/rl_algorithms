# PPO implementation. 


## Python file descriptions

### ppo.py
PPO implementation with clipping objective.
[link to paper](https://arxiv.org/abs/1707.06347)
We use a simple feed forward network [located in](models/actor_critic.py)
equipped with 

### gae.py
Implementation of general advantage estimation.
[link to paper](https://arxiv.org/abs/1506.02438)

Default hyper parameters are located in ppo_hyperparams.py