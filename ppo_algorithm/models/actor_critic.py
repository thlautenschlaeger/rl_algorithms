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


def init_weights_old(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
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
                # nn.Linear(num_hidden_neurons, num_hidden_neurons),
                # nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.Tanh(),
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
                # nn.Linear(num_hidden_neurons, num_hidden_neurons),
                # nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, 1)
            )

            # applies fn recursively to every submodule
            self.apply(init_weights)

        self.no = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):

        mean = self.actor(x)
        value = self.critic(x)
        std = (self.log_std).exp()

        distribution = Normal(mean, std)

        return distribution, value

class ActorCriticMLPShared(nn.Module):
    """
    Single neural network for policy and value with shared parameters
    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, network_type='feed_forward', std=1.0):
        super(ActorCriticMLPShared, self).__init__()

        if network_type == 'feed_forward':

            self.network = nn.Sequential(
                nn.Linear(num_inputs, num_hidden_neurons),
                nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                nn.Tanh(),
                # nn.Linear(num_hidden_neurons, num_hidden_neurons),
                # nn.LayerNorm(num_hidden_neurons, num_hidden_neurons),
                # nn.Tanh(),
                nn.Linear(num_hidden_neurons, num_outputs+1)
            )

            self.apply(init_weights)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):

        output = self.network(x)
        if len(output.shape) < 2:
            output = output.unsqueeze(0)

        mean = output[:, 0:self.num_outputs].clone()
        value = output[:, self.num_outputs:self.num_outputs+1].clone()

        std = torch.clamp((self.log_std).exp(), 0.3)
        distribution = Normal(mean, std)

        return distribution, value


class ActorCriticMLPShared_(nn.Module):
    """
    Single neural network acting policy and value network with shared parameters
    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, std=1.0):
        super(ActorCriticMLPShared_, self).__init__()
        self.hidden_layers = nn.ModuleList()


        self.input_layer = nn.ModuleList([nn.Linear(num_inputs, num_hidden_neurons[0]),
                            nn.LayerNorm(num_hidden_neurons[0], num_hidden_neurons[0]),
                            nn.Tanh()])
        if len(num_hidden_neurons) > 1:
            for i in range(len(num_hidden_neurons) -1):
                self.hidden_layers.append(
                    nn.ModuleList([nn.Linear(num_hidden_neurons[i], num_hidden_neurons[i + 1]),
                     nn.LayerNorm(num_hidden_neurons[i+1], num_hidden_neurons[i+1]),
                     nn.Tanh()])
                )



        self.out_mean = nn.Linear(num_hidden_neurons[-1], num_outputs)
        self.out_value = nn. Linear(num_hidden_neurons[-1], 1)

        self.step_size = len(self.input_layer)
        self.num_hidden_layers = len(self.hidden_layers)


        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)


    def forward(self, x):

        for i in range(self.step_size):
            x = self.input_layer[i](x)

        for i in range(self.num_hidden_layers):
            for j in range(self.step_size):
                x = self.hidden_layers[i][j](x)

        mean = self.out_mean(x)
        std = self.log_std.exp()
        # std = torch.clamp(std, 0.4, 3.)
        value = self.out_value(x)

        dist = Normal(mean, std)

        return dist, value


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






