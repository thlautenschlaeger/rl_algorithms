ppo_params = {
    'advantage_discount' : 0.97, #94
    'bias_var_trade_off' : 0.96, # lambda in advantage
    'clipping' : 0.1,
    # 'clipping' : lambda f: f * 0.2,
    'critic_loss_coeff' :  0.5,
    'entropy_loss_coeff' : 0.00001, # 0.001 for cartpole, 0.000001 for qube
    'max_grad_norm' : .5,
    'minibatch_size' : 32, #4
    'actor_network_std' : 0.88, #0.88

    'num_actors': 1,
    'num_hidden_neurons' : [64, 64],
    'num_hidden_layers' : 1,
    'num_iterations' : 1000000,
    'ppo_epochs' : 8,#3
    'trajectory_size' : 1100,
    'optim_lr' : 0.001,
    # 'optim_lr' : lambda f: f * 1.e-4,

    'visualize' : True,
    'plot_reward' : True,
    'plot_reward_axis' : 1200,
    'plot_time_axis' : 200
}