import torch
import numpy as np

def save_model(network, optimizer, train_rewards, eval_rewards, state_normalizer, epoch, path):
    torch.save({
        'model_state_dict' : network.state_dict(),
        'model' : network,
        'optimizer_state_dict' : optimizer.state_dict(),
        'optimizer' : optimizer,
        'train_rewards' : train_rewards,
        'eval_rewards' : eval_rewards,
        'state_normalizer' : state_normalizer,
        'epoch' : epoch
    }, path+'/save_file.pt')
    # torch.save(network, path+'/ppo_network.pt')
    # torch.save(network.state_dict(),path+'/ppo_network_state_dict.pt' )
    # torch.save(state_normalizer, path+'/normalizer.pt')
    # torch.save(optimizer, path+'/optimizer.pt')
    # torch.save(optimizer.state_dict(), path + '/optimizer_state_dict.pt')
    # np.save(path +'/cum_train_rewards.npy', train_rewards)
    # np.save(path +'/cum_eval_rewards.npy', eval_rewards)

def load_model(path, model, optimizer, from_checkpoint=False):
    if from_checkpoint == True:
        path = path + '/checkpoint'
    checkpoint = torch.load(path+'/save_file.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_rewards = checkpoint['train_rewards']
    eval_rewards = checkpoint['eval_rewards']
    epoch = checkpoint['epoch']
    return model, optimizer, train_rewards, eval_rewards, epoch

    # if from_checkpoint == True:
    #     path = path +'/checkpoint'
    # model.load_state_dict(torch.load(path + '/ppo_network_state_dict.pt'))
    # optimizer.load_state_dict(torch.load(path + '/optimizer_state_dict.pt'))
    # t_rewards = np.load(path+'/cum_train_rewards.npy')
    # e_rewards = np.load(path+'/cum_eval_rewards.npy')
    # return model, optimizer, t_rewards, e_rewards