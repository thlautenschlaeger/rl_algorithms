import gym
from ppo_algorithm.ppo import run_ppo
from ppo_algorithm.ppo_hyperparams import ppo_params


def choose_environment(selection=0):
	""" This method returns a selected environment.
		0 = Cartpole
		1 = Qube
		2 = Levitation
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
	environment = choose_environment(0)

	run_ppo(environment, training_iterations=ppo_params['num_iterations'], num_actors=ppo_params['num_actors'],
			ppo_epochs=ppo_params['ppo_epochs'], trajectory_size=ppo_params['trajectory_size'],
			vis=True, plot=True)
