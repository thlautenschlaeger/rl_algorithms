import gym
from quanser_robots import GentlyTerminating
from ppo_algorithm.ppo import run_ppo


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
	environment = choose_environment(3)
	run_ppo(environment, training_iterations=100000, num_actors=1, ppo_epochs=4, trajectory_size=20, vis=False, plot=True)
