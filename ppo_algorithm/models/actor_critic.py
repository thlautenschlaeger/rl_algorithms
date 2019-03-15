import torch
from torch import nn

def init_weights(m):
    """
    weight initialization

    :param m: network modules
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.0)

class ActorCriticMLPShared(nn.Module):
    """
    Feed forward network trained to predict
    policy parameters and value function estimates.

    """

    def __init__(self, num_inputs, num_hidden_neurons, num_outputs, layer_norm=False, std=1.0):
        """
        initializes network graph

        :param num_inputs: corresponds to state dimension
        :param num_hidden_neurons: list of hidden neurons. each entry
                        corresponds to number of neurons per layer.
        :param num_outputs: corresponds to action dimension
        :param layer_norm: checks if layer_norm should be used
        :param std: policy stddev as network parameter
        """
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
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * torch.tensor(std).log())
        self.apply(init_weights)

    def forward(self, x):

        for _, layer in enumerate(self.net):
            x = layer(x)

        mean = self.out_mean(x)
        std = self.log_std.exp()
        value = self.out_value(x)

        return mean, std, value