from ppo_algorithm.ppo_hyperparams import ppo_params

def arg_parser():
    """
    Empty argument parser
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def ppo_args_parser():
    """
    Parser for PPO
    """

    parser = arg_parser()
    parser.add_argument('--env', help='environment ID',
                        type=str, default='Qube-v0')
    parser.add_argument('--ppoepochs', help='number of ppo optimization epochs',
                        type=int, default=ppo_params['ppo_epochs'])
    parser.add_argument('--training_steps', help='number of total training steps',
                        type=int, default=ppo_params['num_iterations'])
    parser.add_argument('--horizon', help='number of action steps per training epoch',
                        type=int, default=ppo_params['horizon'])
    parser.add_argument('--hneurons',
                        help='number of hidden neurons in list (e.g. [64, 64] for two layers each 64 neurons',
                        type=list, default=ppo_params['num_hidden_neurons'])
    parser.add_argument('--std', help='standard deviation of stochastic policy network',
                        type=float, default=ppo_params['policy_std'])
    parser.add_argument('--minibatches', help='number of minibatches for ppo optimization',
                        type=float, default=ppo_params['minibatches'])
    parser.add_argument('--lam', help='variance bias trade-off for general advantage estimation',
                        type=float, default=ppo_params['lambda'])
    parser.add_argument('--gamma', help='discount factor for general advantage estimation',
                        type=float, default=ppo_params['gamma'])
    parser.add_argument('--cliprange', help='clipping factor for ppo optimaztion',
                        type=float, default=ppo_params['cliprange'])
    parser.add_argument('--vfc', help='value function coefficient',
                        type=float, default=ppo_params['vf_coef'])
    parser.add_argument('-entc', help='entropy coefficient',
                        type=float, default=ppo_params['entropy_coef'])
    parser.add_argument('--evaluate_policy', help='evaluate trained policy from given folder',
                        type=bool, default=False)
    parser.add_argument('--lr', help='training learn rate',
                        type=float, default=ppo_params['lr'])
    parser.add_argument('--layer_norm', help='Layer normalization for actor-critic network',
                        type=bool, default=ppo_params['layer_norm'])
    parser.add_argument('--max_grad_norm', help='sets range for gradient update and clips',
                        type=float,default=ppo_params['max_grad_norm'])
    parser.add_argument('--nevals', help='number of policy evaluations after training policy n steps',
                        type=int, default=ppo_params['num_evals'])
    parser.add_argument('--estep', help='number of policy optimizations until policy gets evaluated',
                        type=int, default=ppo_params['eval_step'])
    parser.add_argument('--path', help='model checkpoint path', type=str, default=None)

    parser.add_argument('--resume', help='continue training boolean flag',
                        type=bool, default=False)
    parser.add_argument('--benchmark', help='if benchmark policy, path has to be provided',
                        type=bool, default=False)
    parser.add_argument('--vis', help='if active, benchmark gets visualized',
                        type=bool, default=False)
    parser.add_argument('--benchsteps', help='number of benchmark evaluations',
                        type=bool, default=False)

    return parser