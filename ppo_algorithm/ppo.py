import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
from ppo_algorithm.utilities.gae import compute_gae
from ppo_algorithm.models import actor_critic
import matplotlib.pyplot as plt

def run_ppo(env, training_iterations, num_actors, ppo_epochs, trajectory_size, vis=False):
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
    num_hidden_neurons = 64

    actor_net = actor_critic.Actor(num_inputs, num_hidden_neurons, num_outputs)
    critic_net = actor_critic.Critic(num_inputs, num_hidden_neurons)


    lel_rerwards = []
    state = env.reset()
    for epoch in range(training_iterations):
        rewards = []
        values = []
        states = []
        actions = []
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
                dist = actor_net(state)
                action = dist.sample()
                value = critic_net(state)

                # execute action
                next_state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])

                log_prob = dist.log_prob(action)
                # save values and rewards for gae
                values.append(value)
                rewards.append(reward)
                old_log_probs.append(log_prob)
                states.append(state)
                actions.append(action)

                state = next_state
                lel_rerward += reward

                if vis: env.render()
                if done:
                    state = env.reset()
                    break

            print('LEL REWARD von ACOTR:', lel_rerward)
            lel_rerwards.append(lel_rerward)

            last_value = critic_net(torch.FloatTensor(next_state))

            advantage_estimates = compute_gae(rewards, values, last_value, 0.99, 0.95)

            values = torch.cat(values).detach()
            old_log_probs = torch.cat(old_log_probs).detach()
            actions = torch.cat(actions)

            # advantage_estimates = torch.cat(advantage_estimates)

            ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, actor_net, critic_net)

            if epoch % 20 == 0:
                plt.plot(lel_rerwards)
                plt.show()



def ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs, actor_net, critic_net, eps=0.2):
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
    :return:
    """
    optim_critic = optim.Adam(actor_net.parameters(), lr=0.001)
    optim_actor = optim.Adam(critic_net.parameters(), lr=0.001)
    # constants for surrogate objective
    c1,c2 = 1, 1


    for _ in range(ppo_epochs):
        for i in range(len(states)):
            dist = actor_net(states[i])
            target_value = critic_net(states[i])
            new_log_prob = dist.log_prob(actions[i])
            entropy = dist.entropy().mean()

            ratio = (new_log_prob - old_log_probs[i]).exp()


            ratio_adv = ratio * advantage_estimates[i]
            clipped_ratio = torch.clamp(ratio, 1-eps, 1+eps) * advantage_estimates[i]

            surrogate1 = torch.min(ratio_adv, clipped_ratio).mean()
            surrogate2 = ((values[i] - target_value) ** 2).mean()

            loss = surrogate1 - c1 * surrogate2 + c2 * entropy

            optim_actor.zero_grad()
            optim_critic.zero_grad()
            loss.backward()
            optim_actor.step()
            optim_critic.step()