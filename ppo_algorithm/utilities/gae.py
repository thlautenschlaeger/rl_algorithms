import torch
import numpy as np

def compute_gae_old(rewards, values, last_value, masks, discount, lamb):
    """
    This method computes general advantage estimates

    :param rewards: list of rewards. size equals length of trajectory
    :param values: list of values from value network
    :param discount: discount factor for gae
    :param lamb: bias variance trade-off. if 1 then high variance if 0 strong bias
    :return: list of advantage estimates
    """
    n = len(rewards)
    values = values.copy()
    values.append(last_value)
    trade_off = discount * lamb
    advantage_estimates = np.zeros(shape=n)
    deltas = np.empty(shape=n)

    for i in range(n):
        deltas[i] = rewards[i] + discount * values[i+1].cpu().detach().numpy() * \
                    masks[i] - values[i].cpu().detach().numpy()

    current_first_index = 0
    for i in range(n):
        advantage = 0
        for j in range(current_first_index, n):
            advantage += deltas[j] * np.power(trade_off * masks[j], j-current_first_index)
        advantage_estimates[i] = advantage
        current_first_index += 1

    return advantage_estimates

def compute_gae(rewards, values, last_value, masks, discount, lamb):
    """
    This method computes general advantage estimates

    :param rewards: list of rewards. size equals length of trajectory
    :param values: list of values from value network
    :param discount: discount factor for gae
    :param lamb: bias variance trade-off. if 1 then high variance if 0 strong bias
    :return: list of advantage estimates
    """
    n = len(rewards)
    values = values.copy()
    values.append(last_value)
    trade_off = discount * lamb
    advantage_estimates = np.zeros(shape=n)
    returns = np.empty(shape=n)
    advantage = 0

    for i in reversed(range(n)):
        delta = rewards[i] + discount * values[i+1].cpu().detach().numpy() * \
                    masks[i] - values[i].cpu().detach().numpy()
        advantage = delta + trade_off * masks[i] * advantage
        advantage_estimates[i] = advantage
        returns[i] = advantage + values[i].cpu().detach().numpy()

    return advantage_estimates, returns
