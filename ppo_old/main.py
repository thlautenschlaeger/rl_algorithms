import gym
from quanser_robots import GentlyTerminating
from ppo_old.ppo_methods import run_ppo


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

	else:
		return gym.make('Pendulum-v0')



if __name__ == '__main__':
	environment = choose_environment(3)
	run_ppo(100, environment, trajectory_size=200, vis=True, plot_reward=True)
