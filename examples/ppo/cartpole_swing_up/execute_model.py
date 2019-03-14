import torch
from ppo_algorithm.ppo import PPO
import gym
from quanser_robots import GentlyTerminating
import os
from torch.distributions import Normal

path = os.path.dirname(__file__)

def load_model(env, path):
	hyper_params = torch.load(path + '/hyper_params.pt', map_location='cpu')
	policy = PPO(env, path, hyper_params).ac_net
	checkpoint = torch.load(path + '/model/save_file.pt', map_location='cpu')
	policy.load_state_dict(checkpoint['model_state_dict'])
	return policy

if __name__ == "__main__":

    env = GentlyTerminating(gym.make('CartpoleSwingShort-v0'))
    model = load_model(env=env, path=path)

    state = env.reset()
    done = False

    while not done:
        mean, _, _ = model(torch.FloatTensor(state))
        dist = Normal(mean, 0)
        action = dist.sample().cpu().detach().numpy()
        state, reward, done, _ = env.step(action)
        env.render()

        print(state, action, reward)

    env.close()

