ppo_params = {
    'advantage_discount' : 0.99,
    'bias_var_trade_off' : 0.97, # lambda for advantage
    'clipping' : 0.02,
    # 'clipping' : lambda f: f * 0.2,
    'critic_loss_coeff' :  1.,
    'entropy_loss_coeff' : 0.000001, # 0.001 for cartpole, 0.000001 for qube
    'max_grad_norm' : 0.5,
    'minibatch_size' : 32, #4
    'min_std' : 0.1,
    'max_std' : 9,
    'actor_network_std' : 1.0,

    'num_actors': 1,
    'num_hidden_neurons' : 64,
    'num_hidden_layers' : 1,
    'num_iterations' : 1000000,
    'ppo_epochs' : 3,#3
    'trajectory_size' : 1024,
    'optim_lr' : 0.0002,
    # 'optim_lr' : lambda f: f * 1.e-4,

    'visualize' : False,
    'plot_reward' : True,
    'plot_reward_axis' : 1200,
    'plot_time_axis' : 200
}