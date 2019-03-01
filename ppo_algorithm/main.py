import gym
# from ppo_algorithm.ppo import run_ppo_old
from ppo_algorithm.ppo import PPO
from ppo_algorithm.ppo_hyperparams import ppo_params
import numpy as np


def choose_environment(selection=0):
	""" This method returns a selected environment.
		0 = Cartpole
		1 = Qube
		2 = Levitation
		3 = Pendulum
	:param selection: select environment as integer
	"""
	if selection == 0:
		return gym.make('CartpoleSwingShort-v0')

	if selection == 1:
		return gym.make('Qube-v0')

	if selection == 2:
		return gym.make('Levitation-v0')

	if selection == 3:
		return gym.make('Pendulum-v0')

	else:
		raise NotImplementedError



if __name__ == '__main__':
	env = choose_environment(3)
	ppo = PPO(env, 100000, 1, ppo_params['ppo_epochs'], ppo_params['trajectory_size'], hidden_neurons=ppo_params['num_hidden_neurons'],
			  std=ppo_params['actor_network_std'], batch_size=ppo_params['minibatch_size'])
	ppo.run_ppo()
	# run_ppo_old(env, training_iterations=ppo_params['num_iterations'], num_actors=ppo_params['num_actors'],
	# 			ppo_epochs=ppo_params['ppo_epochs'], trajectory_size=ppo_params['trajectory_size'],
	# 			vis=ppo_params['visualize'], plot=ppo_params['plot_reward'])
