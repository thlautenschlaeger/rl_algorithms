import torch
import numpy as np

def save_model(network, optimizer, rewards, render_rewards, state_normalizer, path):
    torch.save(network, path+'/ppo_network.pt')
    torch.save(network.state_dict(),path+'/ppo_network_state_dict.pt' )
    torch.save(state_normalizer, path+'/normalizer.pt')
    torch.save(optimizer, path+'/optimizer.pt')
    torch.save(optimizer.state_dict(), path + '/optimizer_state_dict.pt')
    np.save(path+'/cum_train_rewards.npy', rewards)
    np.save(path+'/cum_eval_rewards.npy', render_rewards)

def load_model(path, model, optimizer, from_checkpoint=False):
    if from_checkpoint == True:
        path = path +'/checkpoint'
    model.load_state_dict(torch.load(path + '/ppo_network_state_dict.pt'))
    optimizer.load_state_dict(torch.load(path + '/optimizer_state_dict.pt'))
    t_rewards = np.load(path+'/cum_train_rewards.npy')
    e_rewards = np.load(path+'/cum_eval_rewards.npy')
    return model, optimizer, t_rewards, e_rewards