import gym
from ppo_algorithm.ppo import PPO
import numpy as np
import os
import datetime
import sys
from ppo_algorithm.utils import cmd_util
import torch
from torch.distributions import Normal
from quanser_robots import GentlyTerminating

def run(args=None):
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

def resume_training(env, path):
	"""

	:param env: gym environment
	:param path: stored model path

	"""
	hyper_params = torch.load(path+'/hyper_params.pt')
	print("Training starts")
	ppo = PPO(env, path=path, hyper_params=hyper_params, continue_training=True)
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


def benchmark_policy(env, policy, num_evals, vis=False):
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
		while not done:
			state = torch.FloatTensor(state)
			mean, std, _ = policy(torch.FloatTensor(state))
			dist = Normal(mean, std*0)
			action = dist.sample()
			state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
			cum_reward += reward
			if vis:
				env.render()
		print('Reward:{}, {}'.format(cum_reward, i))
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
	run(sys.argv)