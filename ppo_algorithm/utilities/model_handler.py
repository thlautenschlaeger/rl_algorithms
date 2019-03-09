import torch
import numpy as np

def save_model(model, optimizer, train_rewards, eval_rewards, state_normalizer, epoch, entropy, path):
    torch.save({
        'model_state_dict' : model.state_dict(),
        'model' : model,
        'optimizer_state_dict' : optimizer.state_dict(),
        'optimizer' : optimizer,
        'train_rewards' : train_rewards,
        'eval_rewards' : eval_rewards,
        'state_normalizer' : state_normalizer,
        'epoch' : epoch,
        'entropy' : entropy
    }, path+'/save_file.pt')


def load_model(path, model, optimizer, from_checkpoint=False):
    if from_checkpoint == True:
        path = path + '/checkpoint'
    checkpoint = torch.load(path+'/save_file.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_rewards = checkpoint['train_rewards']
    eval_rewards = checkpoint['eval_rewards']
    epoch = checkpoint['epoch']
    entropy = checkpoint['entropy']
    return model, optimizer, train_rewards, eval_rewards, epoch, entropy