import gym
from quanser_robots import GentlyTerminating
from rs_algorithm.utils import cmd_util
import numpy as np
from rs_algorithm.rs import RandomSearch
import rs_algorithm.rs as rs
import os
import datetime
import matplotlib.pyplot as plt
import torch
import sys



def choose_environment(selection=0):
    if selection == 0:
        return gym.make('CartpoleSwingShort-v0')
    if selection == 1:
        return gym.make('Qube-v0')
    if selection == 2:
        return gym.make('Levitation-v1')
    if selection == 3:
        env = GentlyTerminating(gym.make('CartpoleSwingRR-v0'))
        env.action_space.high = np.array([6.0])
        env.action_space.low = np.array([-6.0])
        return env
    else:
        return gym.make('Pendulum-v0')


def train_rs_policy_v1(rs_hyperparams, env):
    path = os.path.dirname(os.path.abspath(__file__)) + '/data/' + env.unwrapped.spec.id + '_v1_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    random_search = RandomSearch(env, rs_hyperparams, path)
    random_search.ars_v1()


def train_rs_policy_v1_rff(rs_hyperparams, env):
    path = os.path.dirname(os.path.abspath(__file__)) + '/data/' + env.unwrapped.spec.id + '_v1_rff_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    random_search = RandomSearch(env, rs_hyperparams, path)
    random_search.ars_v1_ff()


def train_rs_policy_v2(rs_hyperparams, env):
    path = os.path.dirname(os.path.abspath(__file__)) + '/data/' + env.unwrapped.spec.id + '_v2_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    rs = RandomSearch(env, rs_hyperparams, path)
    rs.ars_v2()

def run_rs(args=None):
    """
    Initializes random search with given arguments. Use default values if not provided

    :param args: parameter dictionary
    """
    parser = cmd_util.rs_args_parser()
    args = parser.parse_known_args(args)[0]
    env = GentlyTerminating(gym.make(args.env))
    rs_params = load_input_to_dict(args)

    if args.resume:
        if args.path != None:
            resume_rs(env, args.path)
        else:
            print("Path not provided")

    if not args.resume:
        if args.path == None:
            path = os.path.dirname(os.path.abspath(__file__)) + '/data/rs_' + env.unwrapped.spec.id + '_' + \
                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            path = args.path

        checkpoint_path = path + '/checkpoint'
        best_policy_path = path + '/best_policy'

        os.makedirs(checkpoint_path)
        os.makedirs(best_policy_path)

        torch.save(rs_params, path+'/hyper_params.pt')

        with open(path+ '/info.txt', 'w') as f:
            print(rs_params, file=f)

        rs = RandomSearch(env, hyperparams=rs_params, path=path)

        if args.version == 0:
            print("Start training augmented random search v1")
            rs.ars_v1()

        elif args.version == 1:
            print("Start training augmented random search v1 with random fourier features")
            rs.ars_v1_ff()
        elif args.version == 2:
            print("Start training augmented random search v2")
            rs.ars_v2()
        else:
            print("Version not available")

def resume_rs(env, path):
    """

    :param env:
    :param path:

    """



def load_input_to_dict(args):
    """
    Loads command line input to dictionary for PPO hyper parameters
    :param args: command line input
    :return: dictionary for hyper parameters
    """
    rs_params = {
        'num_deltas' : args.ndeltas,
        'num_iterations' : args.training_steps,
        'horizon' : args.horizon,
        'lr' : args.lr,
        'bbest' : args.bbest,
        'termination_criterion' : args.tcriterion,
        'num_features' : args.nfeatures,
        'eval_step' : args.estep,
        'sample_noise' : args.snoise
    }
    return rs_params



def plot_graph(list_of_rewards):
    distance_between_datapoints = 20
    means = []
    stds = []
    for i in range(len(list_of_rewards[0])):
        values = []
        for j in range(len(list_of_rewards)):
            values.append(list_of_rewards[j][i])
        means.append(np.mean(values))
        stds.append(np.std(values))

    x = np.arange(0, distance_between_datapoints * len(list_of_rewards[0]), distance_between_datapoints)

    plt.figure()
    plt.errorbar(x, means, xerr=0.0, yerr=stds, fmt='-o')
    plt.title("LOl1")
    plt.ylabel('Expected Return')
    plt.xlabel('Policy Updates')
    plt.show()


if __name__ == '__main__':
    env = choose_environment(0)
    run_rs(sys.argv)
    # path=os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v2_' + \
    #        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # os.makedirs(path)

    # lel.ars_v2_bf(10000, env, 8, 0.025, 0.015, 2, 10, 0, vis=False)
    # lel.ars_v2(300, env, 8, 0.025, 0.015, 2, 1000, path, vis=False)
    # train_rs_policy_v1(rs_params, env)
    # collect_data_v2(rs_params, env, 10)
    # train_rs_policy_v1_rff(rs_params, env)
    # M = rs.load_policy_v1('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_2019-03-12_19-23-42')
    # rs.evaluate_policy_v1(10000, env, M,True)
    # linear_policy, features = rs.load_policy_v1_rff('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_rff_2019-03-11_21-09-25')
    # rs.evaluate_policy_v1_rff(10000, env, linear_policy, features, render=True)

    # M, sigma_rooted, mean = rs.load_policy_v2('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v2_2019-03-11_00-37-31')
    # rs.evaluate_policy_v2(env, M, sigma_rooted, mean, 10000, render=True)

    # list_of_rewards = np.load('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_collected_rewards_2019-03-13_04-07-36/list_of_rewards.npy')
    # plot_graph(list_of_rewards)

