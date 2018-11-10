import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
	""" This class provides a value network for 
	"""

	def __init__(self, num_inputs):
		super(ValueNetwork, self).__init__()
		self.h_layer1 = nn.Linear(num_inputs, 32)
		self.h_layer2 = nn.Linear(32, 32)
		self.out_layer = nn.Linear(32, 1)
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

	def __init__(self, num_inputs, num_outputs):
		super(PolicyNetwork,self).__init__()
		self.h_layer1 = nn.Linear(num_inputs, 32)
		self.h_layer2 = nn.Linear(32, 32)
		self.action_distribution = nn.Linear(32, num_outputs)
		self.action_distribution.weight.data.mul_(0.1)
		self.action_distribution.bias.data.mul_(0.0)



