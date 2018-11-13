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

def compute_general_advantage_estimate(rewards, values, next_value, gamma, lamb):
	""" This method computes the general advantage estimate.
		lamb=1 high variance. Adjusting lamb adjusts bias, variance trade-off
	:param rewards: list of rewards per trajectory
	:param values: list of values computed by critic for previous states
	:param next_value: next value computed by critic for next state
	:param gamma: determines scale of value function
	:param lamb: discount factor for reward shaping
	"""
	gae = 0
	#values = torch.cat((values, next_value)) # do this outside of function
	values = values.copy()
	values.append(next_value)
	advantage_estimates = torch.ones(()).new_empty((1, len(rewards)))[0]
	#adv = np.empty((1, len(rewards)))
	for t_step in range(len(rewards)):
		discount = (gamma * lamb) ** t_step
		delta = rewards[t_step] + gamma * values[t_step + 1] - values[t_step]
		gae = delta + discount * gae # not sure if correct
		advantage_estimates[t_step] = gae + values[t_step]

	return advantage_estimates




def discount_sum_rewards():
	""" This method computes the discounted sum of rewards """
	return



#value_net = ValueNetwork(2, 32)

#inp = np.array([0., 3.])
#inp = torch.tensor([3.4, 2.3])

#print(value_net.forward(inp))
