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
    parser.add_argument('--env', help='environment ID', type=str, default='Qube-v0')
    parser.add_argument('--ppoepochs', help='number of ppo optimization epochs', type=int, default=5)
    parser.add_argument('--ntraining_steps', help='number of total training steps', type=int, default=100)
    parser.add_argument('--horizon', help='number of action steps per training epoch', type=int, default=1024)
    parser.add_argument('--hneurons',
                        help='number of hidden neurons in list (e.g. [64, 64] for two layers each 64 neurons',
                        type=list, default=[64, 64])
    parser.add_argument('--std', help='standard deviation of stochastic policy network', type=float, default=0.88)
    parser.add_argument('--minibatches', help='number of minibatches for ppo optimization', type=float, default=8)
    parser.add_argument('--lam', help='variance bias trade-off for general advantage estimation',
                        type=float, default=0.95)
    parser.add_argument('--gamma', help='discount factor for general advantage estimation',
                        type=float, default=0.99)
    parser.add_argument('--cliprange', help='clipping factor for ppo optimaztion', type=float, default=0.2)
    parser.add_argument('--vfc', help='value function coefficient', type=float, default=0.5)
    parser.add_argument('-entropy_coef', help='entropy coefficient', type=float, default=0.001)
    parser.add_argument('--evaluate_policy', help='evaluate trained policy from given folder',
                        type=bool, default=False)
    parser.add_argument('--path', help='path of previously trained policy. only important if '
                                       '--evaluate_policy True', type=str, default=None)
    parser.add_argument('--lr', help='training learn rate', type=float, default=0.001)
    parser.add_argument('--max_grad_norm', help='sets range for gradient update and clips', type=float,default=0.5)

    return parser