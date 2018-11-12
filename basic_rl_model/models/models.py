import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
	""" This class provides a value network for
		Value network equals the critic
	"""

	def __init__(self, num_inputs, num_hidden_neurons):
		super(ValueNetwork, self).__init__()
		self.h_layer1 = nn.Linear(num_inputs, num_hidden_neurons)
		self.h_layer2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
		self.out_layer = nn.Linear(num_hidden_neurons, 1)
		self.out_layer.weight.data.mul_(0.1)
		self.out_layer.bias.data.mul_(0.0)

	def forward(self, x):
		""" This method runs forward propagation on the above initialized neural network
			and returns output of neural network.

		:param x: input as for neural network. Has dimension of num_inputs
		"""
		x = F.relu(self.h_layer1(x))
		x = F.relu(self.h_layer2(x)) 

		state_values = self.out_layer(x)
		return state_values

class PolicyNetwork(nn.Module):
	""" Policy network equals actor """

	def __init__(self, num_inputs, num_outputs, num_hidden_neurons):
		super(PolicyNetwork,self).__init__()
		self.h_layer1 = nn.Linear(num_inputs, num_hidden_neurons)
		self.h_layer2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
		self.action_distribution = nn.Linear(num_hidden_neurons, num_outputs)
		self.action_distribution.weight.data.mul_(0.1) # not necessary. makes weights smaller
		self.action_distribution.bias.data.mul_(0.0) # set bias to 0


	def forward(self, x):

		x = F.tanh(self.h_layer1(x))
		x = F.tanh(self.h_layer2(x))

		action_mean = self.action_distribution(x)

		return

class ActorCriticNetwork(nn.Module):

	def __init__(self, num_inputs, num_outputs, num_hidden_neurons):
		super(ActorCriticNetwork, self).__init__()
		self.h_layer1 = nn.Linear(num_inputs, num_hidden_neurons)
		self.h_layer2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
		self.action_distribution = nn.Linear(num_hidden_neurons, num_outputs)
		self.action_distribution.weight.data.mul_(0.1) # not necessary. makes weights smaller
		self.action_distribution.bias.d1ata.mul_(0.0) # set bias to 0


	def forward(self, x):

		x = F.tanh(self.h_layer1(x))
		x = F.tanh(self.h_layer2(x))

		action_mean = self.action_distribution(x)

		return

value_net = ValueNetwork(2, 32)

#inp = np.array([0., 3.])
inp = torch.tensor([3.4, 2.3])

print(value_net.forward(inp))
