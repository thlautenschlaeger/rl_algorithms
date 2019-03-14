ppo_params = {
    'gamma' : 0.95, # advantage discount
    'lambda' : 0.95, # bias variance trade-off
    'cliprange' : 0.2,
    'vf_coef' :  0.5,
    'entropy_coef' : 0.1,
    'max_grad_norm' : 0.5,
    'minibatches' : 32, #4
    'policy_std' :  20.0, #0.88
    'num_hidden_neurons' : [32, 32],
    'num_iterations' : 20000,
    'ppo_epochs' : 5,
    'horizon' : 1024,
    'lr' : 1e-3,
    'num_evals' : 1,
    'eval_step' : 50,

    'layer_norm' : False
}

ppo_qube_params = {
    'gamma' : 0.95, # advantage discount
    'lambda' : 0.97, # bias variance trade-off
    'cliprange' : 0.2,
    'vf_coef' :  0.5,
    'entropy_coef' : 0.01,
    'max_grad_norm' : 0.5,
    'minibatches' : 32,
    'policy_std' :  1.0,
    'num_hidden_neurons' : [64, 64],
    'num_iterations' : 1000,
    'ppo_epochs' : 5,
    'horizon' : 1024,
    'lr' : 1e-3,
    'num_evals' : 1,
    'eval_step' : 50,

    'layer_norm' : False
}

ppo_cartpole_params = {
    'gamma' : 0.95, # advantage discount
    'lambda' : 0.97, # bias variance trade-off
    'cliprange' : 0.2,
    'vf_coef' :  0.5,
    'entropy_coef' : 0.01,
    'max_grad_norm' : 0.5,
    'minibatches' : 128,
    'policy_std' :  1.0,
    'num_hidden_neurons' : [64, 64],
    'num_iterations' : 1000,
    'ppo_epochs' : 5,
    'horizon' : 4000,
    'lr' : 1e-3,
    'num_evals' : 1,
    'eval_step' : 50,

    'layer_norm' : False
}