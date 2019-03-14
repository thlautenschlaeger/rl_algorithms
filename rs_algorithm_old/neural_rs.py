import torch
from torch import nn
import numpy as np
from quanser_robots import GentlyTerminating
from ppo_algorithm.normalizer import Normalizer
import matplotlib.pyplot as plt
import torch.distributions as dist

class FeedForward(nn.Module):

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        super(FeedForward, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(num_inputs, num_hidden_neurons),
            nn.Linear(num_hidden_neurons, num_outputs)
        )

    def forward(self, obs):
        out = self.network(obs)

        return out


def sort_max_index_reversed(arr):
    """ This method returns sorted index and compares content of array
    :param arr: array which gets sorted
    """
    n = len(arr)
    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = max(arr[i])

    return np.argsort(tmp)[::-1]

def neural_ars(epochs, env_platform, N, alpha, v, b, H, vis=False):
    """
    This method runs augmented random search v1 on gym environments

    :param epochs: number of iterations
    :param env_platform: gym platform to run algorithm on
    :param N: number of directions sampled per iteration
    :param alpha: stepsize
    :param v: standard deviation of the exploration noise
    :param b: number of top performing directions to use
    :param H: length of trajectories
    """
    cum_reward = []
    env = GentlyTerminating(env_platform)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    normalizer = Normalizer(num_states)

    dim_param = (num_actions, num_states)

    rs_net = FeedForward(num_inputs=num_states, num_hidden_neurons=32, num_outputs=num_actions)

    lel = rs_net.parameters()

    kek = rs_net.state_dict()['network.0.weight'].shape


    # for param in lel:
    #     lel = param.data


    for epoch in range(epochs):
        deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))

        # left negative v1, right positive v1
        total_rewards = np.empty((N, 2))

        for k in range(N):
            # adapt current policy for evaluation
            mj_plus = (linear_policy + v * deltas[k])
            mj_minus = (linear_policy - v * deltas[k])

            total_reward_plus, v1_plus_sequence = perform_rollouts_v2(H, env, mj_plus, normalizer, vis)
            total_reward_minus, v1_minus_sequence = perform_rollouts_v2(H, env, mj_minus, normalizer, vis)

            total_rewards[k] = (total_reward_plus, total_reward_minus)

        sorted_ids = sort_max_index_reversed(total_rewards)

        sum_b_best_rewards = 0
        b_best_rewards = np.empty((2*b))

        for i in range(b):
            id = sorted_ids[i]
            sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
            b_best_rewards[2*i] = total_rewards[id][0]
            b_best_rewards[2*i+1] = total_rewards[id][1]

        std_rewards = np.std(b_best_rewards)
        linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards

        best_total_reward = max(total_rewards[0])
        print('std:',std_rewards)


        print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
        if epoch % 1 == 0:
            cr = evaluate_policy_v2(env, linear_policy, normalizer, vis=False)
            cum_reward.append(cr)
            if cr > 1:
                for i in range(20):
                    evaluate_policy_v2(env, linear_policy, normalizer, vis=True)
                plt.plot(cum_reward)
                plt.show()
        print("-----------------------------------------")

def update_network(network, deltas):
    return network

def sample_deltas(network, num_deltas):
    return


def manipulate_weights_bias(network, name):
    """

    :param network:
    :return:
    """
    net_dict = network.state_dict()
    pos = 0
    sampler = dist.Normal(0, 1)
    for i in range(0, len(net_dict), 2):
        weight = net_dict[name+'.'+i+'weight']
        bias = net_dict[name+'.'+i+'bias']

        delta_weight = sampler.sample(sample_shape=weight.shape)
        delta_bias = sampler.sample(sample_shape=bias.shape)

        net_dict[name+ '.' + i + 'weight'].copy_(weight + delta_weight)
        net_dict[name+ '.' + i + 'bias'].copy_(bias + delta_bias)





def perform_rollouts_v2(H, env, linear_policy, normalizer, vis):
    """
    performs a rollout of given rollout length

    :param H: rollout length
    :param env: gym environment
    :param linear_policy: linear policy
    :param vis: boolean if visualize rendering
    :param sig: covariance
    :param mu: mean of states
    :return: total reward of rollout
    """
    total_reward = 0
    action_reward_sequence = []
    obs = env.reset()

    for i in range(H):
        normalizer.observe(obs)
        obs = normalizer.normalize(obs)

        action = np.dot(linear_policy, obs)
        next_state, reward, done, _ = env.step(action)

        action_reward_sequence.append([action, reward])

        obs = next_state
        total_reward += reward
        # if vis:
        # 	env.render()
        if done:
            obs = env.reset()
    return total_reward, action_reward_sequence

def evaluate_policy_v2(env, policy, normalizer, vis=False):
    state = env.reset()
    cum_reward = 0
    done = False
    while not done:
        # normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy @ state
        state, reward, done, _ = env.step(action)
        cum_reward += reward
        if vis:
            env.render()
    print("cumulative reward: {}".format(cum_reward))
    return cum_reward