import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from rs_algorithm import rs_methods
import numpy as np
import matplotlib.pyplot as plt
from rs_algorithm import basis_functions


def sort_max_index_reversed(arr):
	""" This method returns sorted index and compares content of array
	:param arr: array which gets sorted
	"""
	n = len(arr)
	tmp = np.empty(n)
	for i in range(n):
		tmp[i] = max(arr[i])

	return np.argsort(tmp)[::-1]

def state_normalizer():

def normalize(mean, variance):
	"""

	:param variance:
	:return:
	"""
	std = np.sqrt(variance)
	return



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


def ars_v1_bf(epochs, env_platform, N, alpha, v, b, H, vis=False):
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
		linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards

		print('std:',std_rewards)
		std_list.append(std_rewards)

		print("Best total reward: {} in epoch: {}".format(best_total_reward, epoch))
		if epoch % 10 == 0:
			cr = evaluate_policy_bf(env,policy=linear_policy, bf=bf)
			cum_reward.append(cr)
			plt.plot(cum_reward)
			plt.show()
		print("-----------------------------------------")


def ars_v2(epochs, env_platform, N, alpha, v, b, H, vis=False):
	"""
	This method runs augmented random search on gym environments

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
	show = 0

	#env = GentlyTerminating(gym.make('Qube-v0'))
	env = GentlyTerminating(env_platform)
	num_state_params = env.observation_space.shape[0]
	num_action_params = env.action_space.shape[0]

	dim_param = (num_action_params, num_state_params)

	sig = np.identity(num_state_params)
	mu = np.zeros(num_state_params)
	linear_policy = np.zeros(dim_param)


	highest_reward = 0
	# done = False
	for epoch in range(epochs):
		# env.reset()
		done = False
		# while not done:
		deltas = np.random.standard_normal(size=(N, dim_param[0], dim_param[1]))


		total_rewards = np.empty((N, 2)) # left negative v1, right positive v1

		for k in range(N): # range of delta length parallelize from here
			mj_plus = (linear_policy + v * deltas[k]) # adapt current policy for evaluation
			mj_minus = (linear_policy - v * deltas[k])

			total_reward_plus, v1_plus_sequence, tmp_mu_plus, tmp_cov_plus = perform_rollouts_v2(H, env, mj_plus, sig, mu, vis)
			total_reward_minus, v1_minus_sequence, tmp_mu_minus, tmp_cov_minus = perform_rollouts_v2(H, env, mj_minus, sig, mu, vis)



			total_rewards[k] = (total_reward_minus, total_reward_plus)

		sorted_ids = sort_max_index_reversed(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))
		total_reward = 0
		kek = np.empty(b)

		for i in reversed(range(b)):
			id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[id][0] - total_rewards[id][1]) * deltas[id]
			b_best_rewards[2*i] = total_rewards[id][0]
			b_best_rewards[2*i+1] = total_rewards[id][1]
			kek[i] = total_rewards[id][0] - total_rewards[id][1]

			total_reward += max(total_rewards[id])

		std_rewards = np.std(b_best_rewards)
		# std_rewards = np.std(kek)
		print('std:',std_rewards)
		std_list.append(std_rewards)
		if len(std_list) % 40 == 0:
			plt.plot(std_list)
			plt.show()
		linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards
		# sig =

		print("Total reward:", total_reward)
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

def perform_rollouts_v2(H, env, linear_policy, sig, mu, vis):
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
	tmp_mu_sum = 0
	tmp_cov_sum = 0
	action_reward_sequence = []
	obs = env.reset()
	tmp_mu_sum += obs
	tmp_cov_sum += obs * obs[:,np.newaxis]

	for i in range(H):

		term_1 = np.diag(sig)**(-1/2)
		obs = np.matmul(term_1, (obs - mu))

		action = np.dot(linear_policy, obs)
		next_state, reward, done, _ = env.step(action)

		action_reward_sequence.append([action, reward])

		obs = next_state
		total_reward += reward

		tmp_mu = tmp_mu_sum / i
		tmp_cov = (tmp_cov_sum - tmp_mu * tmp_mu[:, np.newaxis]) / i

		if vis:
			env.render()
		if done:
			state = env.reset()
	return total_reward, action_reward_sequence, tmp_mu, tmp_cov

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

def evaluate_policy_bf(env, policy, bf):
	state = env.reset()
	cum_reward = 0
	done = False
	while not done:
		action = policy @ bf.evaluate(state)
		state, reward, done, _ = env.step(action[0])
		cum_reward += reward
		env.render()
	print("cumulative reward: {}".format(cum_reward))
	return cum_reward



















