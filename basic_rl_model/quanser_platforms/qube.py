import gym
from quanser_robots import GentlyTerminating

env = GentlyTerminating(gym.make('Qube-v0'))
# ctrl =  f: s->a

env.reset()

env.render()
done = False

while not done:
	env.render()
    obs, rwd, done, info = env.step(ctrl)

    done = False


"""
for i in range(1000):
	env.render()
	obs, rwd, done, info = env.step(env.action_space.sample())
"""
