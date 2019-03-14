import torch

def compute_gae(rewards, values, last_value, masks, discount, lamb):
    """
    computes general advantage estimates.

    :param rewards: list of rewards. size equals length of trajectory
    :param values: list of values from value network
    :param last_value: value from last state. necessary for bootstrapping
    :param masks: list containing 0, and 1. 0 sets trajectory to end and
    computes next advantages for next trajectory
    :param discount: discount factor for gae
    :param lamb: bias variance trade-off. if 1 then high variance if 0 strong bias

    :return: advantage estimates and Q-values
    """
    n = len(rewards)
    values = torch.cat((values, last_value))
    trade_off = torch.tensor(discount * lamb)
    advantage_estimates = torch.zeros(size=(n, 1))
    returns = torch.empty(size=(n,1))
    advantage = 0

    for i in reversed(range(n)):
        delta = torch.tensor(rewards[i]) + discount * values[i+1] * masks[i] - values[i]
        advantage = delta + trade_off * masks[i] * advantage
        advantage_estimates[i] = advantage
        returns[i] = advantage + values[i]

    return advantage_estimates, returns
