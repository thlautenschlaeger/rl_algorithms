import gym
# from ppo_algorithm.ppo import run_ppo_old
from ppo_algorithm.ppo import PPO
from ppo_algorithm.ppo_hyperparams import ppo_params
import numpy as np
import os
import datetime
from ppo_algorithm.ppo_hyperparams import ppo_params
import sys
from ppo_algorithm.utilities import cmd_util
import torch



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


def start_ppo(env, args):
	"""
	Initializes PPO object and starts training
	:param env:
	:param args:
	"""
	parser = cmd_util.ppo_args_parser()
	args = parser.parse_known_args(args)[0]
	ppo_params = load_input_to_dict(args)

	path = os.path.dirname(__file__) + '/data/' + env.unwrapped.spec.id + '_' + \
		   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs(path)

	ppo = PPO(env, hyper_params=ppo_params, path=path)
	ppo.run_ppo()

def continue_training(env, path):
	try:
		params = torch.load(path+'/ppo_hyperparams.pt')
	except:
		print("Hyper parameters not found")
		cmd = input("Continue training with default hyper parameters? y [yes], n [no]")

	if cmd == 'y' or cmd == 'yes':
		print("Training starts")
		ppo = PPO(env, path=path, continue_training=True)
		ppo.run_ppo()
	else:
		print("Not continuing training")

def load_input_to_dict(args):
	"""
	Loads command line input to dictionary for PPO hyper parameters
	:param args: command line input
	:return: dictionary for hyper parameters
	"""
	ppo_params = {
		'ppoepochs' : args.ppoepochs,
		'ntrainings_steps' : args.ntraining_steps,
		'nsteps' : args.nsteps,
		'hidden_neurons' : args.hneurons,
		'policy_std' : args.std,
		'minibatches' : args.minibatches,
		'lambda' : args.lam,
		'gamma' : args.gamma,
		'cliprange' : args.cliprange,
		'vf_coef' : args.vfc,
		'entropy_coef' : args.entropy_coef,
		'lr' : args.lr
	}
	return ppo_params


def benchmark_policy(env, path):
	"""
	Loads policy from path and evaluates it

	:param env: gym environment
	:param path: path of policy location
	"""
	return




if __name__ == '__main__':

	env = choose_environment(1)
	ppo = PPO(env, 100000, 1, ppo_params['ppo_epochs'], ppo_params['trajectory_size'], hidden_neurons=ppo_params['num_hidden_neurons'],
			  policy_std=ppo_params['actor_network_std'], minibatches=ppo_params['minibatch_size'])
	ppo.run_ppo()
	# run_ppo_old(env, training_iterations=ppo_params['num_iterations'], num_actors=ppo_params['num_actors'],
	# 			ppo_epochs=ppo_params['ppo_epochs'], trajectory_size=ppo_params['trajectory_size'],
	# 			vis=ppo_params['visualize'], plot=ppo_params['plot_reward'])
