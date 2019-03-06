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
from torch.distributions import Normal
from quanser_robots import GentlyTerminating
from ppo_algorithm.models import actor_critic
import matplotlib.pyplot as plt



def choose_environment(selection=0):
	""" This method returns a selected environment.
		0 = Cartpole
		1 = Qube
		2 = Levitation
		3 = Pendulum
	:param selection: select environment as integer
	"""
	if selection == 0:
		env = GentlyTerminating(gym.make('CartpoleSwingShort-v0'))
		env.action_space.high = np.array([6.0])
		env.action_space.low = np.array([-6.0])

		env.unwrapped.timing.render_rate = 100
		env.observation_space.high[0] *= 0.9
		env.observation_space.low[0] *= 0.9
		return env

	if selection == 1:
		return GentlyTerminating(gym.make('Qube-v0'))

	if selection == 2:
		return GentlyTerminating(gym.make('Levitation-v0'))

	if selection == 3:
		return GentlyTerminating(gym.make('Pendulum-v0'))
	if selection ==4:
		return GentlyTerminating(gym.make('QubeRR-v0'))

	else:
		raise NotImplementedError


def start_ppo(env, args=None):
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
	checkpoint_path = path + '/checkpoint'
	best_policy_path = path + '/best_policy'
	os.makedirs(checkpoint_path)
	os.makedirs(best_policy_path)
	torch.save(ppo_params, path+'/hyper_params.pt')


	ppo = PPO(env, hyper_params=ppo_params, path=path)
	ppo.run_ppo()

def continue_training(env, path):
	try:
		hyper_params = torch.load(path+'/hyper_params.pt')
	except:
		print("Hyper parameters not found")
		cmd = input("Continue training with default hyper parameters? y [yes], n [no]")
	cmd = 'y'
	if cmd == 'y' or cmd == 'yes':
		print("Training starts")
		ppo = PPO(env, path=path, hyper_params=hyper_params, continue_training=True)
		ppo.run_ppo()
	else:
		print("Training not continued")

def load_input_to_dict(args):
	"""
	Loads command line input to dictionary for PPO hyper parameters
	:param args: command line input
	:return: dictionary for hyper parameters
	"""
	ppo_params = {
		'ppo_epochs' : args.ppoepochs,
		'num_iterations' : args.ntraining_steps,
		'horizon' : args.horizon,
		'num_hidden_neurons' : args.hneurons,
		'policy_std' : args.std,
		'max_grad_norm' : args.max_grad_norm,
		'minibatches' : args.minibatches,
		'lambda' : args.lam,
		'gamma' : args.gamma,
		'cliprange' : args.cliprange,
		'vf_coef' : args.vfc,
		'entropy_coef' : args.entropy_coef,
		'lr' : args.lr
	}
	return ppo_params


def benchmark_policy(env, policy, num_evals):
	"""
	Loads policy from path and evaluates it

	:param env: gym environment
	:param path: path of policy location
	:param num_evals: number of policy evaluations
	"""
	reward_list = []
	for i in range(num_evals):
		cum_reward = 0
		done = False
		state = env.reset()
		while not done:
			state = torch.FloatTensor(state)
			mean, std, _ = policy(torch.FloatTensor(state))
			# lel = Normal(dist.mean, dist.stddev*0.)
			dist = Normal(mean, std)
			action = dist.sample()
			state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
			cum_reward += reward
		# env.render()


		reward_list.append(cum_reward)
	# print(cum_reward)
	# plt.plot(reward_list)
	# plt.show()

	print('||||||||||||||||||||||||||||||')
	print('Average Reward:', np.array(reward_list).sum()/num_evals)
	print('||||||||||||||||||||||||||||||')


def load_policy_from_checkpoint(env, path):
	hyper_params = torch.load(path+'/hyper_params.pt', map_location='cpu')
	policy = PPO(env, path, hyper_params).ac_net
	policy.load_state_dict(torch.load(path+'/checkpoint/ppo_network_state_dict.pt', map_location='cpu'))
	return policy

def load_best_policy(env, path):
	hyper_params = torch.load(path + '/hyper_params.pt', map_location='cpu')
	policy = PPO(env, path, hyper_params).ac_net
	policy.load_state_dict(torch.load(path + '/best_policy/ppo_network_state_dict.pt', map_location='cpu'))
	return policy



if __name__ == '__main__':
	# torch.save(ppo_params, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/good_qube_policy_2/hyper_params.pt')
	env = choose_environment(1)
	# policy = load_best_policy(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/good_qube_policy_2')
	# benchmark_policy(env, policy, 5)
	start_ppo(env)
	# policy = load_policy_from_checkpoint(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/test')
	# benchmark_policy(env, policy, 5)

	### for eval
	# continue_training(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/test')

	# env = choose_environment(0)
	# ppo = PPO(env, 100000, 1, ppo_params['ppo_epochs'], ppo_params['trajectory_size'], hidden_neurons=ppo_params['num_hidden_neurons'],
	# 		  policy_std=ppo_params['actor_network_std'], minibatches=ppo_params['minibatch_size'])
	# ppo.run_ppo()
	# benchmark_policy(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/good_qube_policy_2/ppo_network.pt')



	# run_ppo_old(env, training_iterations=ppo_params['num_iterations'], num_actors=ppo_params['num_actors'],
	# 			ppo_epochs=ppo_params['ppo_epochs'], trajectory_size=ppo_params['trajectory_size'],
	# 			vis=ppo_params['visualize'], plot=ppo_params['plot_reward'])
