import torch
import numpy as np


def save_model_ars_v1(policy, rewards, path):
    """

    :param policy:
    :param rewards:
    :param path:
    :return:
    """
    np.save(path + '/M.npy', policy)
    np.save(path + '/training_rewards.npy', rewards)

def save_model_ars_v2(policy, sigma, state_mean, eval_rewards, path):
    """

    :param policy:
    :param sigma:
    :param state_mean:
    :param eval_rewards:

    """
    np.save(path + '/linear_policy.npy', policy)
    np.save(path + '/sigma.npy', sigma)
    np.save(path + '/state_mean.npy', state_mean)
    np.save(path + '/eval_rewards.npy', eval_rewards)

def save_model_ars_v1_rff(policy, features, eval_rewards, path):
    """

    :param policy:
    :param features:
    :param eval_rewards:

    """
    np.save(path + '/linear_policy.npy', policy)
    np.save(path + '/features.npy', features)
    np.save(path + '/eval_rewards.npy', eval_rewards)



def load_model(path, model, optimizer, from_checkpoint=False):
    if from_checkpoint == True:
        path = path +'/checkpoint'
    model.load_state_dict(torch.load(path + '/ppo_network_state_dict.pt'))
    optimizer.load_state_dict(torch.load(path + '/optimizer_state_dict.pt'))
    # t_rewards = np.load(path+'/cum_train_rewards.npy')
    # e_rewards = np.load(path+'/cum_eval_rewards.npy')
    return model, optimizer


def load_policy_v1_rff(path):
    linear_policy = np.load(path+'/linear_policy.npy')
    features = np.load(path+'/features.npy')

    return linear_policy, features

def load_policy_v1(path):
    linear_policy = np.load(path+'/M.npy')
    return linear_policy


def load_policy_v2(path):
    M = np.load(path+'/M.npy')
    sigma_rooted = np.load(path+'/sigma_rooted.npy')
    mean = np.load(path + '/mean.npy')
    return M, sigma_rooted, mean