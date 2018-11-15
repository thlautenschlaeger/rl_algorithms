import torch
import numpy as np
import gym
from quanser_robots import GentlyTerminating


def ars(alpha, N, v, b, env, H):
    """
    :param alpha: ﻿step-size α
    :param N: ﻿number of directions sampled per iteration
    :param v: ﻿standard deviation of the exploration noise
    :param b: ﻿number of top-performing directions to use (b < N is allowed only for V1-t and V2-t)
    :param env: environment of gym
    :return:
    """
    n = env.observation_space.shape[0]
    p = env.action_space.shape[0]
    M, mu, sigma_0, j = initialize(n, p)
    list_of_all_rollouts = list()
    iteration = 0
    while ending_condition_satisfied(iteration):
        iteration = iteration + 1
        deltas = [np.random.standard_normal((p, n)) for _ in range(N)]
        for k in range(N):  # range of delta length
            M_plus = (M + v * deltas[k])
            M_minus = (M - v * deltas[k])
            M_plus_rollout,  M_plus_rollout_reward = calculate_rollout(M_plus, env, H)
            M_minus_rollout,  M_minus_rollout_reward = calculate_rollout(M_minus, env, H)

            list_of_all_rollouts.append([M_plus_rollout_reward, M_plus_rollout, M_minus_rollout, M_minus_rollout_reward, k])

        deltas, list_of_all_rollouts = sort_lists(deltas, list_of_all_rollouts)
        sigma_R = calculate_sigma_R(list_of_all_rollouts, b)
        sum_for_update = calculate_sum_for_update(list_of_all_rollouts, deltas, b, n, p)

        for i in range(len(list_of_all_rollouts[0][1])):
            _, _, _, _ = env.step(list_of_all_rollouts[0][1][i][0])
            env.render()

        M = M + (alpha/(b*sigma_R)) * sum_for_update
        print(M)


def calculate_sigma_R(sorted_list_of_all_rollouts, b):
    list_of_2b_best_rewards = list()
    for i in range(2*b):
        list_of_2b_best_rewards.append(sorted_list_of_all_rollouts[i][0])
        list_of_2b_best_rewards.append(sorted_list_of_all_rollouts[i][3])
    return np.std(np.asarray(list_of_2b_best_rewards))


def initialize(n, p):
    return np.zeros((p, n)), np.zeros(n), np.identity(n), 0


def ending_condition_satisfied(iteration):
    return iteration < 1000


def calculate_sum_for_update(list_of_all_rollouts, deltas, b, n, p):
    sum_lel = np.zeros((p, n))
    for i in range(b):
        sum_lel = sum_lel + ((list_of_all_rollouts[i][0]-list_of_all_rollouts[i][3])*deltas[i])
    return sum_lel


def sort_lists(deltas, list_of_all_rollouts):
    sorted_list_of_all_rollouts = sorted(list_of_all_rollouts, key=lambda x: max(x[0], x[3]), reverse=True)
    sorted_deltas = list()
    for i in range(len(deltas)):
        sorted_deltas.append(deltas[sorted_list_of_all_rollouts[i][4]])
    return deltas, sorted_list_of_all_rollouts


def calculate_rollout(M, env, H):
    state = env.reset()
    list_of_events = list()
    total_reward = 0
    for _ in range(H):
        action = np.matmul(M, state)
        next_state, reward, done, _ = env.step(action)
        list_of_events.append([state, action, reward])
        total_reward = total_reward + reward
        state = next_state
    return list_of_events, total_reward


ars(0.01, 8, 0.03, 4, GentlyTerminating(gym.make('Qube-v0')), 200)