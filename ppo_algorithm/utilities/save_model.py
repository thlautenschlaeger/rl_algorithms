import torch
import numpy as np

def save_files(network, optimizer, rewards, render_rewards, state_normalizer, path):
    torch.save(network, path+'/ppo_network.pt')
    torch.save(state_normalizer, path+'/normalizer.pt')
    torch.save(optimizer, path+'/optimizer.pt')
    np.save(path+'/rewards.npy', rewards)
    np.save(path+'/render_rewards.npy', render_rewards)
