import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from ppo_algorithm.ppo_hyperparams import ppo_params

class Actor(nn.Module):

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, std=0.0):

        super(Actor, self).__init__()

        self.hidden_layer = nn.Linear(num_inputs, num_hidden_neurons)
        self.layer_norm = nn.LayerNorm(num_hidden_neurons, num_hidden_neurons)
        self.out_layer = nn.Linear(num_hidden_neurons, num_outputs)

        # learn optimal standard deviation
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):

        x = self.hidden_layer(x)
        x = F.relu(self.layer_norm(x))
        mean = self.out_layer(x)
        std = self.log_std.exp()

        return Normal(mean, std)

class Critic(nn.Module):

    def __init__(self, num_inputs, num_hidden_neurons):

        super(Critic, self).__init__()

        self.hidden_layer = nn.Linear(num_inputs, num_hidden_neurons)
        self.layer_norm = nn.LayerNorm(num_hidden_neurons, num_hidden_neurons)
        self.out_layer = nn.Linear(num_hidden_neurons, 1)

    def forward(self, x):

        x = self.hidden_layer(x)
        x = F.relu(self.layer_norm(x))
        out_value = self.out_layer(x)

        return out_value


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.0)

class ActorCriticMLP(nn.Module):
    """
    Actor-Critic as multi layer perceptron
    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, network_type='feed_forward', std=1.0):
        super(ActorCriticMLP, self).__init__()

        if network_type == 'feed_forward':

            self.actor = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.ReLU(), nn
                # nn.ELU(),
                nn.Tanh(),
                nn.Linear(num_hidden_neurons, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                nn.Tanh(),
                # nn.Linear(num_hidden_neurons, num_hidden_neurons),
                # nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, num_outputs)
            )

            self.critic = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.ELU(),
                nn.Tanh(),
                nn.Linear(num_hidden_neurons, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                nn.Tanh(),
                # nn.Linear(num_hidden_neurons, num_hidden_neurons),
                # nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, 1)
            )

            # applies fn recursively to every submodule
            self.apply(init_weights)


        elif network_type == 'recurrent':
        #     self.actor == nn.LSTM()
            raise NotImplementedError

        self.no = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):

        mean = self.actor(x)
        value = self.critic(x)
        std = (self.log_std).exp()
        # std = torch.clamp((self.log_std).exp(),ppo_params['min_std'], ppo_params['max_std'])
        # std = ppo_params['actor_network_std']
        distribution = Normal(mean, std)

        return distribution, value

class ActorCriticLSTM(nn.Module):
    """
    Actor-Critic as LSTM Network
    """

    def __init__(self, minibatch_size, num_inputs, num_outputs, num_layers=1):
        super(ActorCriticLSTM, self).__init__()

        self.actor = nn.LSTM(num_inputs, num_outputs, num_layers=num_layers)
        # self.actor_out_layer = nn.Linear(num)
        self.critic = nn.LSTM(num_inputs, 1, num_layers=num_layers)

    # def forward(self, x):






