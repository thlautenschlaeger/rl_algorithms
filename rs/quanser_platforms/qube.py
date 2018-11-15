import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from rs import rs
import numpy as np

def sort_max_index(arr):
	""" This method returns sorted index and compares content of array
	:param arr: array which gets sorted
	"""
	n = len(arr)
	tmp = np.empty(n)
	for i in range(n):
		tmp[i] = max(arr[i])

	return np.argsort(tmp)


def run_rs(epochs, N, alpha, v, b, H, vis=False):
	""" This method runs random search on Qube-v0
	:param epochs: number of iterations
	:param N: number of directions sampled per iteration
	:param alpha: stepsize
	:param v: standard deviation of the exploration noise
	:param b: number of top performing directions to use
	:param H: length of trajectories
	"""
	env = GentlyTerminating(gym.make('Qube-v0'))
	num_states = env.observation_space.shape[0]
	num_actions = env.action_space.shape[0]

	dim_param = (num_actions, num_states)

	state = env.reset()

	M = np.zeros(dim_param)

	done = False
	for _ in range(epochs):
		deltas = [np.random.standard_normal(dim_param) for _ in range(N)]

		total_rewards = np.empty((N, 2)) # left negative v1, right positive v1

		for k in range(N): # range of delta length
			mj_plus = (M + v * deltas[k])

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
		for i in reversed(range(b)):
			sorted_id = sorted_ids[i]
			sum_b_best_rewards += (total_rewards[sorted_id][0] - total_rewards[sorted_id][1]) * deltas[sorted_id]
			b_best_rewards[2*i] = total_rewards[sorted_id][0]
			b_best_rewards[2*i+1] = total_rewards[sorted_id][1]

		#print("Policy Matrix:", M)
		#print("-----------------------------------------")


		std_rewards = np.std(b_best_rewards)

		env.reset()
		for i in range(len(v1_plus_sequence)):
			_, _, _, _ = env.step(v1_plus_sequence[i][0])
			env.render()
		env.reset()
		for i in range(len(v1_minus_sequence)):
			_, _, _, _ = env.step(v1_minus_sequence[i][0])
			env.render()

		M = M + (alpha/(b*std_rewards)) * sum_b_best_rewards

run_rs(500, 16, 0.01, 0.09, 8, 400, vis=True)













