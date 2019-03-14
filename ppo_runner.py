import gym
# from ppo_algorithm.ppo import run_ppo_old
from ppo_algorithm.ppo import PPO
from ppo_algorithm.ppo_hyperparams import ppo_params
import numpy as np
import os
import datetime
from ppo_algorithm.ppo_hyperparams import ppo_params
import sys
from ppo_algorithm.utils import cmd_util
import torch
from torch.distributions import Normal
from quanser_robots import GentlyTerminating
from ppo_algorithm.models import actor_critic
import matplotlib.pyplot as plt
from gym.wrappers import Monitor



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
		env = GentlyTerminating(gym.make('Qube-v0'))
		return env

	if selection == 2:
		env = GentlyTerminating(gym.make('Levitation-v1'))
		return env

	if selection == 3:
		return GentlyTerminating(gym.make('Pendulum-v0'))
	if selection ==4:
		return GentlyTerminating(gym.make('QubeRR-v0'))

	else:
		raise NotImplementedError


def train_ppo(args=None):
	"""
	Initializes PPO object and starts training

	:param env: gym environment
	:param args: arguments for PPO
	"""
	parser = cmd_util.ppo_args_parser()
	args = parser.parse_known_args(args)[0]
	env = GentlyTerminating(gym.make(args.env))
	ppo_params = load_input_to_dict(args)


	if args.resume:
		if args.path != None:
			resume_training(env, args.path)
		else:
			print("Path not provided training not continued")

	if not args.resume:
		if args.path == None:
			path = os.path.dirname(os.path.abspath(__file__)) + '/data/ppo' + env.unwrapped.spec.id + '_' + \
				   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		else:
			path = args.path

		checkpoint_path = path + '/checkpoint'
		best_policy_path = path + '/best_policy'
		os.makedirs(checkpoint_path)
		os.makedirs(best_policy_path)
		torch.save(ppo_params, path+'/hyper_params.pt')

		with open(path+'/info.txt', 'w') as f:
			print(ppo_params, file=f)


		ppo = PPO(env, hyper_params=ppo_params, path=path)
		ppo.run_ppo()

def resume_training(env, path, new=False):
	"""

	:param env: gym environment
	:param path:

	"""
	hyper_params = torch.load(path+'/hyper_params.pt')
	print("Training starts")
	ppo = PPO(env, path=path, hyper_params=hyper_params, continue_training=True, new=new)
	ppo.run_ppo()


def load_input_to_dict(args):
	"""
	Loads command line input to dictionary for PPO hyper parameters
	:param args: command line input
	:return: dictionary for hyper parameters
	"""
	ppo_params = {
		'ppo_epochs' : args.ppoepochs,
		'num_iterations' : args.training_steps,
		'horizon' : args.horizon,
		'num_hidden_neurons' : args.hneurons,
		'policy_std' : args.std,
		'max_grad_norm' : args.max_grad_norm,
		'minibatches' : args.minibatches,
		'lambda' : args.lam,
		'gamma' : args.gamma,
		'cliprange' : args.cliprange,
		'vf_coef' : args.vfc,
		'entropy_coef' : args.entc,
		'lr' : args.lr,
		'num_evals' : args.nevals,
		'eval_step' : args.estep,
		'layer_norm' : args.layer_norm
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
	policy.eval()
	for i in range(num_evals):
		cum_reward = 0
		done = False
		state = env.reset()
		transition_rewards = []
		while not done:
			state = torch.FloatTensor(state)
			mean, std, _ = policy(torch.FloatTensor(state))
			dist = Normal(mean, std*0)
			action = dist.sample()
			# action = torch.clamp(action, min=-6, max=6)
			state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
			cum_reward += reward
			transition_rewards.append(cum_reward)
			env.render()
		print('Reward:{}, {}'.format(cum_reward, i))
		# np.save(path+'/tr/transition_rewards'+str(i)+'.npy', np.array(transition_rewards))
		reward_list.append(cum_reward)

	print('------------------------------')
	print('Average Reward: {}'.format(np.array(reward_list).sum()/num_evals))
	print('------------------------------')


def load_policy_from_checkpoint(env, path):
	"""
	loads policy from last checkpoint

	:param env: gym environment
	:param path: model location
	:return: model from checkpoint
	"""
	hyper_params = torch.load(path+'/hyper_params.pt', map_location='cpu')
	policy = PPO(env, path, hyper_params).ac_net
	checkpoint = torch.load(path+'/checkpoint/save_file.pt', map_location='cpu')
	policy.load_state_dict(checkpoint['model_state_dict'])

	return policy


def load_best_policy(env, path):
	"""
	loads policy that performed best on evaluation

	:param env: gym environment
	:param path: location
	:return: best trained policy
	"""
	hyper_params = torch.load(path + '/hyper_params.pt', map_location='cpu')
	policy = PPO(env, path, hyper_params).ac_net
	checkpoint = torch.load(path + '/best_policy/save_file.pt', map_location='cpu')
	policy.load_state_dict(checkpoint['model_state_dict'])

	return policy



if __name__ == '__main__':
	# torch.save(ppo_params, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/good_qube_policy_2/hyper_params.pt')
	# env = choose_environment(0)
	# env = GentlyTerminating(gym.make('CartpoleStabShort-v0'))
	# policy = load_best_policy(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/good_qube_policy_2')
	# policy = load_policy_from_checkpoint(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/CartpoleSwingShort-v0_2019-03-10_14-14-20')
	# policy = load_policy_from_checkpoint(env,
	# 									 '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/CartpoleSwingShort-v0_2019-03-12_20-50-29')
	# policy = load_policy_from_checkpoint(env, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/CartpoleSwingShort-v0_2019-03-12_22-10-07')

	# path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/' \
	# 	   'reinforcement_learning/project/rl_algorithms/data/3copy'

	# path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/CartpoleSwingShort-v0_2019-03-13_15-05-02'
	# path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/' \
	# 	   'reinforcement_learning/project/rl_algorithms/data/cart3'
	# policy = load_policy_from_checkpoint(env, path)
	# policy = load_best_policy(env, path)

	# benchmark_policy(env, policy, 1)
	train_ppo(sys.argv)

	# start_ppo(args)
