import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from ppo_algorithm.ppo_hyperparams import ppo_params

def init_weights_old(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.0)

class ActorCriticMLPShared(nn.Module):
    """
    Single neural network acting policy and value network with shared parameters
    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, layer_norm=False, std=1.0):
        super(ActorCriticMLPShared, self).__init__()

        if layer_norm:
            self.net = nn.ModuleList([nn.Linear(num_inputs, num_hidden_neurons[0]),
                                  nn.LayerNorm(num_hidden_neurons[0], num_hidden_neurons[0]),
                                  nn.Tanh()])
        else:
            self.net = nn.ModuleList([nn.Linear(num_inputs, num_hidden_neurons[0]),
                                      nn.Tanh()])

        if len(num_hidden_neurons) > 1:
            for i in range(len(num_hidden_neurons) - 1):
                self.net.append(nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]))
                if layer_norm:
                    self.net.append(nn.LayerNorm(num_hidden_neurons[i + 1], num_hidden_neurons[i + 1]))
                self.net.append(nn.Tanh())

        self.out_mean = nn.Linear(num_hidden_neurons[-1], num_outputs)
        self.out_value = nn.Linear(num_hidden_neurons[-1], 1)

        self.step_size = len(self.net)
        self.num_hidden_layers = len(num_hidden_neurons)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)

    def forward(self, x):

        for _, layer in enumerate(self.net):
            x = layer(x)

        mean = self.out_mean(x)
        # std = torch.sigmoid(self.log_std.exp())
        std = self.log_std.exp()
        # std = torch.clamp(std, 0.4)
        value = self.out_value(x)

        # dist = Normal(mean, std)

        return mean, std, value

class ActorCriticMLPShared__(nn.Module):
    """
    Single neural network acting policy and value network with shared parameters
    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, std=1.0):
        super(ActorCriticMLPShared, self).__init__()

        self.net = nn.ModuleList([nn.Linear(num_inputs, num_hidden_neurons[0]),
                                  nn.LayerNorm(num_hidden_neurons[0], num_hidden_neurons[0]),
                                  nn.Tanh()])

        if len(num_hidden_neurons) > 1:
            for i in range(len(num_hidden_neurons) - 1):
                self.net.append(nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]))
                self.net.append(nn.LayerNorm(num_hidden_neurons[i + 1], num_hidden_neurons[i + 1]))
                self.net.append(nn.Tanh())

        self.out_mean = nn.Linear(num_hidden_neurons[-1], num_outputs)
        self.out_value = nn.Linear(num_hidden_neurons[-1], 1)

        self.step_size = len(self.net)
        self.num_hidden_layers = len(num_hidden_neurons)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)

    def forward(self, x):

        for _, layer in enumerate(self.net):
            x = layer(x)

        mean = self.out_mean(x)
        # std = torch.sigmoid(self.log_std.exp())
        std = self.log_std.exp()
        # std = torch.clamp(std, 0.4)
        value = self.out_value(x)

        dist = Normal(mean, std)
        return dist, value

        # return mean, std, value