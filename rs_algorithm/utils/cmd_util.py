from rs_algorithm.rs_hyperparams import rs_params

def arg_parser():
    """
    Empty argument parser
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def rs_args_parser():
    """
    Parser for RS
    """

    parser = arg_parser()

    parser.add_argument('--env', help='environment ID',
                        type=str, default='Qube-v0')

    parser.add_argument('--ndeltas', help='number of random delta samples to '
                                             'update linear policy',
                        type=int, default=rs_params['num_deltas'])

    parser.add_argument('--training_steps', help='number of training epochs',
                        type=int,
                        default=rs_params['num_iterations'])

    parser.add_argument('--lr', help='learn rate for policy update',
                        type=float, default=rs_params['lr'])

    parser.add_argument('--bbest', help='number of best rewards to use to update policy',
                        type=int, default=rs_params['bbest'])

    parser.add_argument('--horizon', help='length of rollout horizon',
                        type=int, default=rs_params['horizon'])

    parser.add_argument('--tcriterion', help='reward as termination criterion',
                        type=float, default=rs_params['termination_criterion'])

    parser.add_argument('--nfeatures', help='number of fourier features',
                        type=int, default=rs_params['num_features'])

    parser.add_argument('--estep', help='policy gets evaluated after every eval_steps',
                        type=int, default=rs_params['eval_step'])

    parser.add_argument('--path', help='path to save checkpoints and model',
                        type=str, default=None)

    parser.add_argument('--snoise', help='sample noise parameter',
                        type=float, default=rs_params['sample_noise'])

    parser.add_argument('--version', help='rs version. 0=arsv1, 1=arsv1rff, 2=arsv2',
                        type=int, default=0)
    parser.add_argument('--resume', help='bool flag that checks if training should be resumed. path must be provided',
                        type=bool, default=False)

    return parser

