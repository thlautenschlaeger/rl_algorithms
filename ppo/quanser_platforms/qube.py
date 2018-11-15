import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from ppo.models.actorcritic import PolicyNetwork
from ppo.models.actorcritic import ValueNetwork
from ppo.models.actorcritic import compute_general_advantage_estimate

from ppo.ppo import ppo_update


def test_env(epochs, name, model, vis=False):
    env = GentlyTerminating(gym.make('Qube-v0')) # Qube-v0 is actual name
    #env = GentlyTerminating(gym.make(name))
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    model = PolicyNetwork(num_inputs, num_outputs, 64) # model
    value_model = ValueNetwork(num_inputs, 64)

    state = env.reset()
    optimizer_p = optim.Adam(model.parameters(), lr= 0.01)
    optimizer_v = optim.Adam(value_model.parameters(), lr = 0.01)

    total_reward = 0
    for i in range(epochs):
        state = torch.FloatTensor(state)
        dist = model.forward(state)
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy())
        loss = -value_model.forward(state) * torch.FloatTensor([reward])
        optimizer_p.zero_grad()
        optimizer_v.zero_grad()
        loss.backward()
        optimizer_p.step()
        optimizer_v.step()
        state = next_state
        total_reward += reward

        print("Total rewards:", reward)
        if vis: env.render()

def run_ppo(epochs, vis=False):
    """ This method computes ppo on
    :param epochs: number of epochs to run ppo
    :param vis: rendering simulation if true
    """
    env = GentlyTerminating(gym.make('Qube-v0'))
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    actor = PolicyNetwork(num_inputs, num_outputs, 64)
    critic = ValueNetwork(num_inputs, 64)

    state = env.reset()

    # initialize optimizers for each network
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)
    done = False

    for _ in range(epochs):
        while not done:

            ppo_epochs = 5
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            trajectory_size = 20
            entropy = 0
            total_reward = 0
            for i in range(trajectory_size):
                state = torch.FloatTensor(state)
                dist = actor.forward(state)
                value = critic.forward(state)

                action = dist.sample()
                next_state, reward, done, info = env.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward))

                states.append(state)

                actions.append(action)

                state = next_state
                total_reward += reward

                if vis: env.render()

            print("Reward:", total_reward)
            print("----------------------")


            next_state = torch.FloatTensor(next_state)
            next_value = critic.forward(next_state)


            advantages_estimates = compute_general_advantage_estimate(rewards, values, next_value, 0.99, 0.95)

            # detach from current graph

            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states).split(num_inputs)
            actions = torch.cat(actions)
            advantage = advantages_estimates - values

            ppo_update(ppo_epochs, states, actions, log_probs, advantages_estimates, advantage, optimizer_actor,
                       optimizer_critic, actor.forward, critic.forward, 0.2)


run_ppo(100, vis=False)
