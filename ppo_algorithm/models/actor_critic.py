import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

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

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, network_type='feed_forward', std=0.0):
        super(ActorCritic, self).__init__()

        if network_type == 'feed_forward':
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, num_outputs)
            )

            self.critic = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, 1)
            )
        # elif network_type == 'recurrent':
        #     self.actor == nn.



        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)


    def forward(self, x):

        mean = self.actor(x)
        value = self.critic(x)
        std = torch.exp(self.log_std)

        distribution = Normal(mean, std)

        return distribution, value




