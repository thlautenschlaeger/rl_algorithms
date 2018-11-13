import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ValueNetwork(nn.Module):
	""" Value network equals the critic """

	def __init__(self, num_inputs, num_hidden_neurons):
		super(ValueNetwork, self).__init__()

		self.hidden_layer1 = nn.Linear(num_inputs, num_hidden_neurons)
		self.hidden_layer2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
		self.out_layer = nn.Linear(num_hidden_neurons, 1)
		self.out_layer.weight.data.mul_(0.1)
		self.out_layer.bias.data.mul_(0.0)

	def forward(self, x):
		""" This method runs forward propagation on the above initialized neural network
			and returns output of neural network.

		:param x: input as for neural network. Has dimension of num_inputs
		"""
		print(x)
		x = torch.tanh(self.hidden_layer1(x))
		x = torch.tanh(self.hidden_layer2(x))

		state_values = self.out_layer(x)
		return state_values

class PolicyNetwork(nn.Module):
	""" Policy network equals actor """

	def __init__(self, num_inputs, num_outputs, num_hidden_neurons, std=0.01):
		""" Initializes neural net with parameters
		:param num_inputs: dimension of observation space
		:param num_outputs: dimension of action space
		:param num_hidden_neurons: number of hidden neurons per layer
		"""
		super(PolicyNetwork,self).__init__()

		self.hidden_layer1 = nn.Linear(num_inputs, num_hidden_neurons)
		self.hidden_layer2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
		self.action_mean = nn.Linear(num_hidden_neurons, num_outputs)
		self.action_mean.weight.data.mul_(0.1) # not necessary. makes weights smaller
		self.action_mean.bias.data.mul_(0.0) # set bias to 0
		self.std = nn.Parameter(torch.ones(1, num_outputs) * std)

		#self.log_std = nn


	def forward(self, x):

		x = torch.tanh(self.hidden_layer1(x))
		x = torch.tanh(self.hidden_layer2(x))

		mu = self.action_mean(x)
		std = self.std.exp()
		dist = Normal(mu, std)

		return dist

def compute_general_advantage_estimate(lamb, discount, traj_length, ):
	""" This method computes the general advantage estimate"""

def discount_sum_rewards():
	""" This method computes the discounted sum of rewards """
	return



#value_net = ValueNetwork(2, 32)

#inp = np.array([0., 3.])
#inp = torch.tensor([3.4, 2.3])

#print(value_net.forward(inp))
