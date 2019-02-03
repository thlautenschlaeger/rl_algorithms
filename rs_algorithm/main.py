import gym
from quanser_robots import GentlyTerminating
from rs_algorithm.rs_methods import ars_v1
from rs_algorithm.rs_methods import ars_v1_bf
from rs_algorithm.rs_methods import ars_v2


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
	env = choose_environment(0)
	# ars_v2(10000, environment, 8, 0.0025, 0.02, 8, 1000, vis=False)
	# ars_v1(epochs=10000, env_platform=env, N=20, alpha=0.025, v=0.02, b=8, H=1024)
	# ars_v1(epochs=10000, env_platform=env, N=20, alpha=0.01, v=0.025, b=4, H=1024)
	ars_v1_bf(epochs=10000, env_platform=env, N=8, alpha=0.025, v=0.015, b=4, H=1024)
	# ars_v1(epochs=10000, env_platform=env, N=8, alpha=0.025, v=0.015, b=4, H=1024)
	# run_rs(10000, environment, 50, 0.025, 0.02, 4, 700, vis=False)
	# run_rs(1000, environment, 16, 0.08, 0.8, 8, 50, vis=True, plot_reward=True)
