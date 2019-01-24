ppo_params = {
    'advantage_discount' : 0.99,
    'bias_var_trade_off' : 0.95,
    'clipping' : 0.02,
    # 'clipping' : lambda f: f * 0.2,
    'critic_loss_coeff' : 0.5,
    'entropy_loss_coeff' : 0.01,
    'max_grad_norm' : 0.5,
    'minibatch_size' : 32, #4
    'actor_network_std' : 0.0,
    'num_actors': 2,
    'num_hidden_neurons' : 128,
    'num_hidden_layers' : 1,
    'num_iterations' : 1000000,
    'ppo_epochs' : 3,#3
    'trajectory_size' : 1024,
    'optim_lr' : 0.00005,
    # 'optim_lr' : lambda f: f * 1.e-4,

    'visualize' : False,
    'plot_reward' : False
}