import torch
import numpy as np

def save_files(network, optimizer, rewards, render_rewards, state_normalizer, path):
    torch.save(network, path+'/ppo_network.pt')
    torch.save(network.state_dict(),path+'/ppo_network_state_dict.pt' )
    torch.save(state_normalizer, path+'/normalizer.pt')
    torch.save(optimizer, path+'/optimizer.pt')
    torch.save(optimizer.state_dict(), path + '/optimizer_state_dict.pt')
    np.save(path+'/rewards.npy', rewards)
    np.save(path+'/render_rewards.npy', render_rewards)

