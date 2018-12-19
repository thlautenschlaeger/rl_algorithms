import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from rs import rs_methods
import numpy as np
import matplotlib.pyplot as plt


def sort_max_index(arr):
	""" This method returns sorted index and compares content of array
	:param arr: array which gets sorted
	"""
	n = len(arr)
	tmp = np.empty(n)
	for i in range(n):
		tmp[i] = max(arr[i])

	return np.argsort(tmp)


def run_rs(epochs, env_platform, N, alpha, v, b, H, vis=False):
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
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	dim_param = (num_actions, num_states)



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

			total_reward_plus, v1_plus_sequence = perform_rollouts(H, env, mj_plus, vis)
			total_reward_minus, v1_minus_sequence = perform_rollouts(H, env, mj_minus, vis)

			total_rewards[k] = (total_reward_minus, total_reward_plus)

		sorted_ids = sort_max_index(total_rewards)

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
		if len(std_list) % 10 == 0:
			plt.plot(std_list)
			plt.show()
		linear_policy = linear_policy + (alpha / ((b * std_rewards))) * sum_b_best_rewards

		print("Total reward:", total_reward)
		print("-----------------------------------------")
		# reward_list.append(total_reward)
		'''
		if total_reward > highest_reward:
			highest_reward = total_reward
			print("New highest reward: {} epoch:{}".format(highest_reward, epoch))

		if epoch % 50 == 0:
			print("current reward: {} epoch: {}".format(total_reward, epoch))
			reward_list.append(total_reward)
			plt.plot(reward_list)
			plt.show()
		'''

def perform_rollouts(H, env, linear_policy, vis):
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
		# action = np.matmul(linear_policy, state)
		action = np.dot(linear_policy, state)
		next_state, reward, done, _ = env.step(action)

		action_reward_sequence.append([action, reward])

		state = next_state
		total_reward += reward
		if vis:
			env.render()
		if done: break
	return total_reward, action_reward_sequence

















