ppo_params = {
    'gamma' : 0.97, # advantage discount
    'lambda' : 0.96, # bias variance trade-off
    'cliprange' : 0.2,
    'vf_coef' :  0.5,
    'entropy_coef' : 0.01, # 0.001 for cartpole, 0.000001 for qube
    'max_grad_norm' : 1.,
    'minibatches' : 64, #4
    'policy_std' :  0., #0.88
    'num_actors': 1,
    'num_hidden_neurons' : [64],
    'num_iterations' : 1000000,
    'ppo_epochs' : 10,
    'horizon' : 3000,
    'lr' : 0.001,
    'layer_norm' : False,

    'visualize' : False,
    'plot_reward' : True
}