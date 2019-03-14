# Group 1 

## How to run algorithms

1. Create virtual environment and install dependencies

```
python3 -m virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

2. Execute algorithm. Example for to execute PPO 
with default hyperparameters
```
python3 run_ppo --env Qube-v0 
```
 
# Project Structure

```
rl_algorithms
├── README.md
├── examples 	<-- example files with pre-trained policies
│   ├── ppo 	
│   │   ├── README.md
│   │   ├── cartpole_swing_up
│   │   │   ├── README.md
│   │   │   ├── execute_model.py
│   │   │   ├── hyper_params.pt
│   │   │   └── model
│   │   │       └── save_file.pt
│   │   ├── levitation
│   │   ├── qube
│   │   │   ├── README.md
│   │   │   ├── execute_model.py
│   │   │   ├── hyper_params.pt
│   │   │   └── model
│   │   │       └── save_file.pt
│   │   └── qube_rr
│   │       ├── README.md
│   │       ├── execute_model.py
│   │       ├── hyper_params.pt
│   │       └── model
│   │           └── save_file.pt
│   └── rs
│       ├── cartpole_swing_up
│       ├── levitation
│       └── qube
├── ppo_algorithm 		<-- ppo implementation
│   ├── README.md
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.py
│   │   ├── gae.py
│   │   ├── normalizer.py
│   │   ├── ppo.py
│   │   └── ppo_hyperparams.py
│   ├── gae.py
│   ├── main.py
│   ├── models				<-- model neural network
│   │   ├── __init__.py
│   │   └── actor_critic.py
│   ├── normalizer.py
│   ├── ppo.py
│   ├── ppo_hyperparams.py
│   └── utilities
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.py
│       │   ├── cmd_util.py 
│       │   ├── model_handler.py
│       │   └── plot_utility.py
│       ├── cmd_util.py
│       ├── model_handler.py
│       └── plot_utility.py
├── ppo_runner.py 		<-- ppo execution from command-line
├── rs_runner.py 		<-- rs execution from command-line
└── rs_algorithm
    ├── README.md
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.py
    │   ├── basis_functions.py
    │   ├── neural_rs.py
    │   ├── normalizer.py
    │   └── rs_methods.py
    ├── basis_functions.py
    ├── main.py
    ├── neural_rs.py
    ├── normalizer.py
    ├── policy.py
    ├── rs_hyperparams.py
    └── rs_methods.py
        
```
