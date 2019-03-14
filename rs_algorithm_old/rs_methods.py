import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from rs_algorithm_old import rs_methods
import numpy as np
import matplotlib.pyplot as plt
from rs_algorithm_old import basis_functions
from rs_algorithm_old.normalizer import Normalizer


def sort_max_index_reversed(arr):
	""" This method returns sorted index and compares content of array
	:param arr: array which gets sorted
	"""
	n = len(arr)
	tmp = np.empty(n)
	for i in range(n):
		tmp[i] = max(arr[i])

	return np.argsort(tmp)[::-1]

# def state_normalizer():

# def normalize(mean, variance):
# 	"""
#
# 	:param variance:
# 	:return:
# 	"""
# 	std = np.sqrt(variance)
# 	return



def ars_v1(epochs, env_platform, N, alpha, v, b, H, vis=False):
	"""
	This method runs augmented random search v1 on gym environments

	:param epochs: number of iterations
	:param env_platform: gym platform to run algorithm on
	:param N: number of directions sampled per iteration
	:param alpha: stepsize
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""
	reward_list = []
	std_list = []
	cum_reward = []
	env = GentlyTerminating(env_platform)
	num_state_params = env.observation_space.shape[0]
	num_action_params = env.action_space.shape[0]

	dim_param = (num_action_params, num_state_params)

	linear_policy = np.zeros(dim_param)
	for epoch in range(epochs):
		deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))

		# left negative v1, right positive v1
		total_rewards = np.empty((N, 2))

		for k in range(N):
			# adapt current policy for evaluation
			mj_plus = (linear_policy + v * deltas[k])
			mj_minus = (linear_policy - v * deltas[k])

			total_reward_plus, v1_plus_sequence = perform_rollouts_v1(H, env, mj_plus, vis)
			total_reward_minus, v1_minus_sequence = perform_rollouts_v1(H, env, mj_minus, vis)

			total_rewards[k] = (total_reward_plus, total_reward_minus)

		sorted_ids = sort_max_index_reversed(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))

		for i in range(b):
			id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
			b_best_rewards[2*i] = total_rewards[id][0]
			b_best_rewards[2*i+1] = total_rewards[id][1]

		std_rewards = np.std(b_best_rewards)
		linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards

		best_total_reward = max(total_rewards[0])
		print('std:',std_rewards)


		print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
		if epoch % 10 == 0:
			cr = evaluate_policy(env, linear_policy)
			cum_reward.append(cr)
			plt.plot(cum_reward)
			plt.show()
		print("-----------------------------------------")


def ars_v1_bf(epochs, env_platform, N, lr, v, b, H, vis=False):
	"""
	This method runs augmented random search v1 on gym environments

	:param epochs: number of iterations
	:param env_platform: gym platform to run algorithm on
	:param N: number of directions sampled per iteration
	:param lr: learnrate
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""
	degree = 2
	std_list = []
	cum_reward = []

	env = GentlyTerminating(env_platform)
	num_state_params = env.observation_space.shape[0]
	num_action_params = env.action_space.shape[0]

	# dim_param = (num_action_params, num_state_params*degree+1)
	dim_param = (num_action_params, np.power(num_state_params,num_state_params)+1)
	# bf = basis_functions.PolynomialBasis(dim_state=num_state_params, degree=degree)
	bf = basis_functions.FourierBasis(state_dim=num_state_params)

	linear_policy = np.zeros(dim_param)
	for epoch in range(epochs):

		deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))
		total_rewards = np.empty((N, 2)) # left negative v1, right positive v1

		for k in range(N):
			mj_plus = (linear_policy + v * deltas[k])
			mj_minus = (linear_policy - v * deltas[k])

			total_reward_plus, v1_plus_sequence = perform_rollouts_v1_bf(H, env, mj_plus, bf,vis)
			total_reward_minus, v1_minus_sequence = perform_rollouts_v1_bf(H, env, mj_minus, bf,vis)

			total_rewards[k] = (total_reward_plus, total_reward_minus)


		sorted_ids = sort_max_index_reversed(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))

		for i in range(b):
			id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
			b_best_rewards[2*i] = total_rewards[id][0]
			b_best_rewards[2*i+1] = total_rewards[id][1]

		best_total_reward = max(total_rewards[0])
		std_rewards = np.std(b_best_rewards)
		linear_policy = linear_policy + (lr / ((b * std_rewards)+1e-12)) * sum_b_best_rewards

		print('std:',std_rewards)
		std_list.append(std_rewards)

		# print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
		if epoch % 1 == 0:
			cr = evaluate_policy_bf(env,policy=linear_policy, bf=bf)
			cum_reward.append(cr)
			print("Cumulative reward {} in epoch: {}".format(cr, epoch))
			if cr > 2000:
				plt.plot(cum_reward)
				plt.show()
				for i in range(20):
					evaluate_policy_bf(env, policy=linear_policy, bf=bf, vis=True)
		print("-----------------------------------------")

def ars_v2(epochs, env_platform, N, alpha, v, b, H, vis=False):
	"""
	This method runs augmented random search v1 on gym environments

	:param epochs: number of iterations
	:param env_platform: gym platform to run algorithm on
	:param N: number of directions sampled per iteration
	:param alpha: stepsize
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""

	cum_reward = []
	env = GentlyTerminating(env_platform)
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	normalizer = Normalizer(num_states)

	dim_param = (num_actions, num_states)

	linear_policy = np.zeros(dim_param)
	for epoch in range(epochs):
		deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))

		# left negative v1, right positive v1
		total_rewards = np.empty((N, 2))

		for k in range(N):
			# adapt current policy for evaluation
			mj_plus = (linear_policy + v * deltas[k])
			mj_minus = (linear_policy - v * deltas[k])

			total_reward_plus, v1_plus_sequence = perform_rollouts_v2(H, env, mj_plus, normalizer, vis)
			total_reward_minus, v1_minus_sequence = perform_rollouts_v2(H, env, mj_minus, normalizer, vis)

			total_rewards[k] = (total_reward_plus, total_reward_minus)

		sorted_ids = sort_max_index_reversed(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))

		for i in range(b):
			id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
			b_best_rewards[2*i] = total_rewards[id][0]
			b_best_rewards[2*i+1] = total_rewards[id][1]

		std_rewards = np.std(b_best_rewards)
		linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards

		best_total_reward = max(total_rewards[0])
		print('std:',std_rewards)


		print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
		if epoch % 1 == 0:
			cr = evaluate_policy_v2(env, linear_policy, normalizer, vis=False)
			cum_reward.append(cr)
			if cr > 1:
				for i in range(20):
					evaluate_policy_v2(env, linear_policy, normalizer, vis=True)
				plt.plot(cum_reward)
				plt.show()
		print("-----------------------------------------")

def ars_v2_bf(epochs, env_platform, N, alpha, v, b, H, vis=False):
	"""
	This method runs augmented random search v1 on gym environments

	:param epochs: number of iterations
	:param env_platform: gym platform to run algorithm on
	:param N: number of directions sampled per iteration
	:param alpha: stepsize
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""

	cum_reward = []
	env = GentlyTerminating(env_platform)
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	bf = basis_functions.FourierBasis(state_dim=num_states)
	normalizer = Normalizer(num_states)

	dim_param = (num_actions, np.power(num_states, num_states)+1)

	linear_policy = np.zeros(dim_param)
	for epoch in range(epochs):
		deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))

		# left negative v1, right positive v1
		total_rewards = np.empty((N, 2))

		for k in range(N):
			# adapt current policy for evaluation
			mj_plus = (linear_policy + v * deltas[k])
			mj_minus = (linear_policy - v * deltas[k])

			total_reward_plus, v1_plus_sequence = perform_rollouts_v2_bf(H, env, mj_plus, normalizer, bf, vis)
			total_reward_minus, v1_minus_sequence = perform_rollouts_v2_bf(H, env, mj_minus, normalizer, bf, vis)

			total_rewards[k] = (total_reward_plus, total_reward_minus)

		sorted_ids = sort_max_index_reversed(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))

		for i in range(b):
			id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
			b_best_rewards[2*i] = total_rewards[id][0]
			b_best_rewards[2*i+1] = total_rewards[id][1]

		std_rewards = np.std(b_best_rewards)
		linear_policy = linear_policy + (alpha / ((b * std_rewards)+1e-12)) * sum_b_best_rewards

		best_total_reward = max(total_rewards[0])
		print('std:',std_rewards)


		print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
		if epoch % 1 == 0:
			cr = evaluate_policy_bf_v2(env, linear_policy, bf, normalizer, vis=False)
			cum_reward.append(cr)
			if cr > 1:
				for i in range(20):
					evaluate_policy_bf_v2(env, linear_policy, bf, normalizer, vis=True)
				plt.plot(cum_reward)
				plt.show()
		print("-----------------------------------------")



def perform_rollouts_v1_bf(H, env, linear_policy, basis_function, vis):
	"""
	performs a rollout of given rollout length

	:param H: rollout length
	:param env: gym environment
	:param linear_policy: linear policy
	:param basis_function: basis function object
	:param vis: boolean if visualize rendering
	:return: total reward of rollout
	"""
	total_reward = 0
	action_reward_sequence = []

	state = env.reset()
	for i in range(H):
		feature = basis_function.evaluate(state)
		action = np.dot(linear_policy, feature)

		next_state, reward, done, _ = env.step(action[0])

		action_reward_sequence.append([action, reward])

		state = next_state
		total_reward += reward
		if vis:
			env.render()
		if done:
			# total_reward = 0
			state = env.reset()
	return total_reward, action_reward_sequence

def perform_rollouts_v1(H, env, linear_policy, vis):
	"""
	performs a rollout of given rollout length

	:param H: rollout length
	:param env: gym environment
	:param linear_policy: linear policy
	:param vis: boolean if visualize rendering
	:return: total reward of rollout
	"""
	total_reward = 0
	action_reward_sequence = []

	state = env.reset()
	for i in range(H):

		action = np.dot(linear_policy, state)
		next_state, reward, done, _ = env.step(action)

		action_reward_sequence.append([action, reward])

		state = next_state
		total_reward += reward
		if vis:
			env.render()
		if done:
			total_reward = 0
			state = env.reset()
	return total_reward, action_reward_sequence

def perform_rollouts_v2(H, env, linear_policy, normalizer, vis):
	"""
	performs a rollout of given rollout length

	:param H: rollout length
	:param env: gym environment
	:param linear_policy: linear policy
	:param vis: boolean if visualize rendering
	:param sig: covariance
	:param mu: mean of states
	:return: total reward of rollout
	"""
	total_reward = 0
	action_reward_sequence = []
	obs = env.reset()

	for i in range(H):
		normalizer.observe(obs)
		obs = normalizer.normalize(obs)

		action = np.dot(linear_policy, obs)
		next_state, reward, done, _ = env.step(action)

		action_reward_sequence.append([action, reward])

		obs = next_state
		total_reward += reward

		# if vis:
		# 	env.render()
		if done:
			obs = env.reset()
	return total_reward, action_reward_sequence

def perform_rollouts_v2_bf(H, env, linear_policy, normalizer, basis_function,vis):
	"""
	performs a rollout of given rollout length

	:param H: rollout length
	:param env: gym environment
	:param linear_policy: linear policy
	:param vis: boolean if visualize rendering
	:param sig: covariance
	:param mu: mean of states
	:return: total reward of rollout
	"""
	total_reward = 0
	action_reward_sequence = []
	obs = env.reset()

	for i in range(H):
		normalizer.observe(obs)
		obs = normalizer.normalize(obs)
		feature = basis_function.evaluate(obs)

		action = np.dot(linear_policy, feature)
		next_state, reward, done, _ = env.step(action[0])

		action_reward_sequence.append([action, reward])

		obs = next_state
		total_reward += reward

		# if vis:
		# 	env.render()
		if done:
			obs = env.reset()
	return total_reward, action_reward_sequence

def evaluate_policy(env, policy):
	state = env.reset()
	cum_reward = 0
	done = False
	while not done:
		action = policy @ state
		state, reward, done, _ = env.step(action)
		cum_reward += reward
		env.render()
	print("cumulative reward: {}".format(cum_reward))
	return cum_reward

def evaluate_policy_v2(env, policy, normalizer, vis=False):
	state = env.reset()
	cum_reward = 0
	done = False
	while not done:
		# normalizer.observe(state)
		state = normalizer.normalize(state)
		action = policy @ state
		state, reward, done, _ = env.step(action)
		cum_reward += reward
		if vis:
			env.render()
	print("cumulative reward: {}".format(cum_reward))
	return cum_reward

def evaluate_policy_bf(env, policy, bf, vis=False):
	state = env.reset()
	cum_reward = 0
	done = False
	while not done:
		action = policy @ bf.evaluate(state)
		state, reward, done, _ = env.step(action[0])
		cum_reward += reward
		if vis:
			env.render()
	print("cumulative reward: {}".format(cum_reward))
	return cum_reward

def evaluate_policy_bf_v2(env, policy, bf, normalizer, vis=False):
	state = env.reset()
	cum_reward = 0
	done = False
	while not done:
		state = normalizer.normalize(state)
		action = policy @ bf.evaluate(state)
		state, reward, done, _ = env.step(action[0])
		cum_reward += reward
		if vis:
			env.render()
	print("cumulative reward: {}".format(cum_reward))
	return cum_reward



















