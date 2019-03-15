import torch
import numpy as np

def save_model(model, optimizer, train_rewards, eval_rewards, eval_rewards_std, epoch, entropy, path):
    """
    saves model as dictionary file

    :param model: actor-critic model
    :param optimizer: pyTorch optimizer
    :param train_rewards: total rollout rewards
    :param eval_rewards: expected evaluation rewards
    :param eval_rewards_std: stddev of expected evaluation rewards
    :param epoch: current training epoch
    :param entropy: list of policy entropies for every iteration
    :param path: path where to put save file
    :return:
    """
    torch.save({
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'train_rewards' : train_rewards,
        'eval_rewards' : eval_rewards,
        'eval_rewards_std' : eval_rewards_std,
        'epoch' : epoch,
        'entropy' : entropy
    }, path+'/save_file.pt')


def load_model(path, model, optimizer, from_checkpoint=False):
    """
    loads model and

    :param path: save file location
    :param model: initialized model
    :param optimizer: initialized optimizer
    :param from_checkpoint:
    :return:
    """
    if from_checkpoint == True:
        path = path + '/checkpoint'


    checkpoint = torch.load(path+'/save_file.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_rewards = checkpoint['train_rewards']
    eval_rewards = checkpoint['eval_rewards']
    eval_rewards_std = checkpoint['eval_rewards_std']
    epoch = checkpoint['epoch']
    entropy = checkpoint['entropy']
    return model, optimizer, train_rewards, eval_rewards, eval_rewards_std, epoch, entropy