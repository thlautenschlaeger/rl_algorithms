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
	""" This method runs random search on Qube-v0
	:param epochs: number of iterations
	:param env_platform: gym platform to run algorithm on
	:param N: number of directions sampled per iteration
	:param alpha: stepsize
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""
	reward_list = []
	show = 0

	#env = GentlyTerminating(gym.make('Qube-v0'))
	env = GentlyTerminating(env_platform)
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	dim_param = (num_actions, num_states)

	state = env.reset()

	linear_policy = np.zeros(dim_param)

	done = False
	for _ in range(epochs):
		deltas = [np.random.standard_normal(dim_param) for _ in range(N)]

		total_rewards = np.empty((N, 2)) # left negative v1, right positive v1

		for k in range(N): # range of delta length parallelize from here
			mj_plus = (linear_policy + v * deltas[k])

			v1_plus_sequence = []
			v1_minus_sequence = []

			total_reward_plus = 0
			total_reward_minus = 0
			state = env.reset()

			for _ in range(H):
				action = np.matmul(mj_plus,state)
				next_state, reward, done, _ = env.step(action)

				v1_plus_sequence.append([action, reward])

				state = next_state
				total_reward_plus += reward
				#if vis: env.render()
				if done: break

			state = env.reset()
			for _ in range(H):
				action = np.matmul(mj_plus, state)
				next_state, reward, done, _ = env.step(action)

				v1_minus_sequence.append([action, reward])

				state = next_state
				total_reward_minus += reward
				#if vis: env.render()
				if done: break

			total_rewards[k] = (total_reward_minus, total_reward_plus)

		sorted_ids = sort_max_index(total_rewards)

		sum_b_best_rewards = 0
		b_best_rewards = np.empty((2*b))
		total_reward = 0
		for i in reversed(range(b)):
			sorted_id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[sorted_id][0] - total_rewards[sorted_id][1]) * deltas[sorted_id]
			b_best_rewards[2*i] = total_rewards[sorted_id][0]
			b_best_rewards[2*i+1] = total_rewards[sorted_id][1]
			total_reward += max(total_rewards[sorted_id])



		std_rewards = np.std(b_best_rewards)
		linear_policy = linear_policy + (alpha / ((b * std_rewards) + 0.00001)) * sum_b_best_rewards

		print("Total reward:", total_reward)
		print("-----------------------------------------")
		reward_list.append(total_reward)

		plt.plot(reward_list)

		show += 1
		if show % 100 == 0:
			plt.pause(0.05)
			show = 0

	if vis:
		env.reset()
		if total_reward_plus > total_reward_minus:

			for i in range(len(v1_plus_sequence)):
				_, _, _, _ = env.step(v1_plus_sequence[i][0])
				env.render()

		else:
			for i in range(len(v1_minus_sequence)):
				_, _, _, _ = env.step(v1_minus_sequence[i][0])
				env.render()



#run_rs(30000, 16, 0.025, 0.02, 8, 500, vis=True)
#run_rs(30000, 16, 0.04, 0.2, 8, 500, vis=True)













