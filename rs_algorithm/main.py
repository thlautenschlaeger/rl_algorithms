import gym
from quanser_robots import GentlyTerminating
from rs_algorithm.neural_rs import neural_ars
import numpy as np
from rs_algorithm.rs import RandomSearch
import rs_algorithm.rs as rs
from rs_algorithm.rs_hyperparams import rs_params
import os
import datetime
import matplotlib.pyplot as plt

import rs_algorithm.rs_methods as lel


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
    path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v1_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    random_search = RandomSearch(env, rs_hyperparams, path)
    random_search.ars_v1()


def train_rs_policy_v1_rff(rs_hyperparams, env):
    path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v1_rff_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    random_search = RandomSearch(env, rs_hyperparams, path)
    random_search.ars_v1_ff()


def train_rs_policy_v2(rs_hyperparams, env):
    path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v2_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    rs = RandomSearch(env, rs_hyperparams, path)
    rs.ars_v2()

def collect_data_v1(rs_hyperparams, env, number_of_runs):
    list_of_rewards = []
    for i in range(number_of_runs):
        path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v1_' + \
               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(path)
        random_search = RandomSearch(env, rs_hyperparams, path)
        rewards = random_search.ars_v1()
        list_of_rewards.append(rewards)
        print('---------------')
    path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v1_collected_rewards' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    np.save(path + '/list_of_rewards.npy', list_of_rewards)

def collect_data_v2(rs_hyperparams, env, number_of_runs):
    list_of_rewards = []
    for i in range(number_of_runs):
        path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v2_' + \
               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(path)
        random_search = RandomSearch(env, rs_hyperparams, path)
        rewards = random_search.ars_v2()
        list_of_rewards.append(rewards)
        print('---------------')
    path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v2_collected_rewards_' + \
           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(path)
    np.save(path + '/list_of_rewards.npy', list_of_rewards)


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
    # path=os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_v2_' + \
    #        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # os.makedirs(path)

    # lel.ars_v2_bf(10000, env, 8, 0.025, 0.015, 2, 10, 0, vis=False)
    # lel.ars_v2(300, env, 8, 0.025, 0.015, 2, 1000, path, vis=False)
    # train_rs_policy_v1(rs_params, env)
    collect_data_v2(rs_params, env, 10)
    # train_rs_policy_v1_rff(rs_params, env)
    # M = rs.load_policy_v1('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_2019-03-12_19-23-42')
    # rs.evaluate_policy_v1(10000, env, M,True)
    # linear_policy, features = rs.load_policy_v1_rff('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_rff_2019-03-11_21-09-25')
    # rs.evaluate_policy_v1_rff(10000, env, linear_policy, features, render=True)

    # M, sigma_rooted, mean = rs.load_policy_v2('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v2_2019-03-11_00-37-31')
    # rs.evaluate_policy_v2(env, M, sigma_rooted, mean, 10000, render=True)

    # list_of_rewards = np.load('/Users/jan/Desktop/rl_algorithms-v0.8/rs_algorithm/data/CartpoleSwingShort-v0_v1_collected_rewards_2019-03-13_04-07-36/list_of_rewards.npy')
    # plot_graph(list_of_rewards)

