import torch
import numpy as np


def save_model(policy, path, hyperparams=0):
    torch.save(hyperparams, path+'/rs_hyperparams.pt')
    torch.save(policy, path+'/policy.pt')


def load_model(path, model, optimizer, from_checkpoint=False):
    if from_checkpoint == True:
        path = path +'/checkpoint'
    model.load_state_dict(torch.load(path + '/ppo_network_state_dict.pt'))
    optimizer.load_state_dict(torch.load(path + '/optimizer_state_dict.pt'))
    # t_rewards = np.load(path+'/cum_train_rewards.npy')
    # e_rewards = np.load(path+'/cum_eval_rewards.npy')
    return model, optimizer