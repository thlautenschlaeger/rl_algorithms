import gym
from quanser_robots import GentlyTerminating
import torch
import algorithms
from algorithms.models.models import PolicyNetwork
from algorithms.models.models import ValueNetwork


def test_env(epochs, name, model, vis=False):
    env = GentlyTerminating(gym.make('Qube-v0')) # Qube-v0 is actual name
    #env = GentlyTerminating(gym.make(name))
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    model = PolicyNetwork(num_inputs, num_outputs, 64).forward # model
    value_model = ValueNetwork(1, 64).forward

    state = env.reset()

    total_reward = 0
    #loss = 0.0
    for i in range(epochs):
        state = torch.FloatTensor(state)
        dist = model(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        loss = -value_model(action) * torch.FloatTensor([reward])
        loss.backward()
        state = next_state
        total_reward += reward

        print("Total rewards:", reward)
        if vis: env.render()


test_env(1000,'kek', 'model', True)


"""
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward
"""