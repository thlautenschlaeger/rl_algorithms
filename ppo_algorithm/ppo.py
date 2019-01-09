import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from ppo_algorithm.utilities.gae import compute_gae
from ppo_algorithm.models import actor_critic
import matplotlib.pyplot as plt
import numpy as np

def run_ppo(env, training_iterations, num_actors, ppo_epochs, trajectory_size, vis=False, plot=False):
    """
    This method runs ppo_algorithm algorithm.

    :param env: gym environment
    :param training_iterations: number of training iterations
    :param ppo_epochs: number of ppo_algorithm epochs
    :param vis: if true than show visualization
    :return:
    """

    env = GentlyTerminating(env)

    # ---------------------------------------- #
    #### initialize actor-critic networks ####
    # ---------------------------------------- #
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    num_hidden_neurons = 32

    # actor_net = actor_critic.Actor(num_inputs, num_hidden_neurons, num_outputs)
    # critic_net = actor_critic.Critic(num_inputs, num_hidden_neurons)

    ac_net = actor_critic.ActorCriticMLP(num_inputs, num_hidden_neurons, num_outputs)


    cumulative_rerwards = []
    state = env.reset()
    for epoch in range(training_iterations):
        rewards = []
        values = []
        states = []
        actions = []
        # used for penalize entering done state
        masks = []
        # necessary for ratio of old and new policy during ppo_algorithm update
        old_log_probs = []

        for i in range(num_actors):

            # ------------------ #
            #### run policy ####
            # ------------------ #
            lel_rerward = 0
            for t in range(trajectory_size):
                state = torch.FloatTensor(state)

                # sample state from normal distribution
                dist, value = ac_net(state)
                action = dist.sample()

                # dist = actor_net(state)
                # value = critic_net(state)

                # execute action
                next_state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])

                log_prob = dist.log_prob(action)
                # save values and rewards for gae
                values.append(value)
                old_log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                state = next_state
                rewards.append(reward)
                masks.append(1 - done)
                lel_rerward += reward

                if vis:
                    env.render()
                if done:
                    state = env.reset()
                    break


            print('LEL REWARD von ACOTR:', lel_rerward)
            cumulative_rerwards.append(lel_rerward)

            _, last_value = ac_net(torch.FloatTensor(next_state))

            advantage_estimates = compute_gae(rewards, values, last_value, masks, 0.99, 0.9)

            values = torch.cat(values).detach()
            old_log_probs = torch.cat(old_log_probs).detach()
            actions = torch.cat(actions)

            # advantage_estimates = torch.cat(advantage_estimates)

            # ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, actor_net, critic_net, 5)
            ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, ac_net, 5, 0.2)

            if plot:
                if epoch % 100 == 0:
                    plot_avg_reward = []
                    for i in range(len(cumulative_rerwards)):
                        tmp = np.mean(cumulative_rerwards[i:i+trajectory_size])
                        plot_avg_reward.append(tmp)

                    plt.plot(plot_avg_reward)
                    plt.show()


# def ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, actor_net, critic_net, minibatch_size, eps=0.1):
def ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, ac_net,
                   minibatch_size, eps=0.2):
    """
    This method performs proximal policy update over batches of inputs

    :param ppo_epochs:
    :param advantage_estimates:
    :param states:
    :param actions:
    :param values:
    :param old_log_probs:
    :param actor_net:
    :param critic_net:
    :param minibatch_size:
    :return:
    """
    # optim_critic = optim.Adam(actor_net.parameters(), lr=0.001)
    # optim_actor = optim.Adam(critic_net.parameters(), lr=0.001)
    ac_optim = optim.Adam(ac_net.parameters(), lr=0.0001)

    # constants for surrogate objective
    c1,c2 = 0.99, 0.01

    randomize = np.arange(len(states))
    # randomize = torch.randperm(len(states))

    # shape states for further processing
    states = torch.stack(states)

    # normalize advantages
    advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / (advantage_estimates.std() + 1e-8)

    for _ in range(ppo_epochs):
        # shuffle inputs every ppo epoch
        np.random.shuffle(randomize)
        old_log_probs = old_log_probs[randomize]
        actions = actions[randomize]
        values = values[randomize]
        advantage_estimates = advantage_estimates[randomize]
        states = states[randomize]

        for i in range(0, len(states), minibatch_size):
            dist, target_value = ac_net(states[i:i+minibatch_size])

            new_log_prob = dist.log_prob(actions[i:i+minibatch_size])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_prob - old_log_probs[i:i+minibatch_size])

            # shape advantage estimate vector
            advantage_batch = torch.FloatTensor(advantage_estimates[i:i+minibatch_size]).view(advantage_estimates[i:i+minibatch_size].shape[0], 1)

            surr = ratio * advantage_batch
            clipped_surr = torch.clamp(ratio, 1-eps, 1+eps) * advantage_batch

            actor_loss = torch.min(surr, clipped_surr).mean() # not sure when to use min or max
            critic_loss = ((advantage_batch - values[i:i+minibatch_size] - target_value) ** 2).mean()
            # critic_loss = ((values[i:i + minibatch_size] - target_value) ** 2).mean()

            # loss = actor_loss + c1 * critic_loss - c2 * entropy

            loss = actor_loss - c1 * critic_loss + c2 * entropy

            ac_optim.zero_grad()
            # computes gradient
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 0.5)
            ac_optim.step()


