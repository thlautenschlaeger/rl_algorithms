ppo_params = {
    'gamma' : 0.98, # advantage discount
    'lambda' : 0.97, # bias variance trade-off
    'cliprange' : 0.2,
    'vf_coef' :  0.5,
    'entropy_coef' : 0.00001, # 0.001 for cartpole, 0.000001 for qube
    'max_grad_norm' : 1.,
    'minibatches' : 32, #4
    'policy_std' :  0.0, #0.88
    'num_actors': 1,
    'num_hidden_neurons' : [128, 128],
    'num_iterations' : 1000000,
    'ppo_epochs' : 10,
    'horizon' : 1100,
    'lr' : 0.001,
    'layer_norm' : True,

    'visualize' : False,
    'plot_reward' : True
}