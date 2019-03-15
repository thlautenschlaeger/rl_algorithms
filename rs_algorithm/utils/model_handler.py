import torch
import numpy as np


def save_model_ars_v1(policy, rewards, path):
    """ saves a policy
    :param policy: the policy
    :param rewards: the rewarads
    :param path: path to save
    """
    np.save(path + '/M.npy', policy)
    np.save(path + '/eval_rewards.npy', rewards)


def load_policy_v1(path):
    """ Loads a policy
    :param path: file to load the path
    """
    linear_policy = np.load(path+'/M.npy')
    rewards = np.load(path + '/training_rewards.npy')
    return linear_policy, rewards


def save_model_ars_v2(policy, sigma, state_mean, eval_rewards, number_of_encountered_states, path):
    """ saves a policy
    :param policy: the policy
    :param sigma: sigma used in policy
    :param state_mean: the mean of states
    :param eval_rewards: all rewards
    :param number_of_encountered_states: number of all encountered states
    :param path: where to save the model

    """
    np.save(path + '/linear_policy.npy', policy)
    np.save(path + '/sigma.npy', sigma)
    np.save(path + '/state_mean.npy', state_mean)
    np.save(path + '/nr_encountered_states.npy', number_of_encountered_states)
    np.save(path + '/eval_rewards.npy', eval_rewards)


def load_policy_v2(path):
    """ Loads a policy
    :param path: path to load from
    """
    M = np.load(path+'/M.npy')
    sigma = np.load(path+'/sigma_rooted.npy')
    mean = np.load(path + '/mean.npy')
    nr_states = np.load(path + '/nr_encountered_states.npy')
    eval_rewards = np.load(path + '/eval_rewards.npy')
    return M, sigma, mean, nr_states, eval_rewards


def save_model_ars_v1_rff(policy, features, eval_rewards, path):
    """
    :param policy: the policy
    :param features: the features
    :param eval_rewards: collected rewards
    :param path: path to save the model
    """
    np.save(path + '/linear_policy.npy', policy)
    np.save(path + '/features.npy', features)
    np.save(path + '/eval_rewards.npy', eval_rewards)


def load_policy_v1_rff(path):
    """ Loads a policy
    :param path: path to load from
    """
    linear_policy = np.load(path+'/linear_policy.npy')
    features = np.load(path+'/features.npy')
    rewards = np.load(path+'/eval_rewards.npy')
    return linear_policy, features, rewards
