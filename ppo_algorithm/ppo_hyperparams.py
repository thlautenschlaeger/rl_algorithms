ppo_params = {
    'advantage_discount' : 0.95, #94
    'bias_var_trade_off' : 0.97, # lambda in advantage
    # 'clipping' : 0.05,
    'clipping' : lambda f: f * 0.2,
    'critic_loss_coeff' :  0.2,
    'entropy_loss_coeff' : 0.0000001, # 0.001 for cartpole, 0.000001 for qube
    'max_grad_norm' : .5,
    'minibatch_size' : 32, #4
    'actor_network_std' :  0.88, #0.88

    'num_actors': 1,
    'num_hidden_neurons' : [64, 64],
    'num_hidden_layers' : 1,
    'num_iterations' : 1000000,
    'ppo_epochs' : 8,#3
    'trajectory_size' : 1100,
    'optim_lr' : 0.001,
    'optim_lrr' : lambda f, x: f * x,

    'visualize' : False,
    'plot_reward' : True,
    'plot_reward_axis' : 1200,
    'plot_time_axis' : 200
}