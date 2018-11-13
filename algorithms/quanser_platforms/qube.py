import gym
from quanser_robots import GentlyTerminating

env = GentlyTerminating(gym.make('Qube-v0'))
# action space has dimensionof 1
# observation space has dimension of 6
# ctrl =  f: s->a
o = env.observation_space.shape[0]
a = env.action_space.shape[0]
env.reset()

env.render()
done = False
a = 0.5
while not done:
    env.render()
    obs, rwd, done, info = env.step(a)

    done = False


"""
for i in range(1000):
	env.render()
	obs, rwd, done, info = env.step(env.action_space.sample())
"""
