from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from ppo_algorithm.gae import compute_gae
from ppo_algorithm.models import actor_critic
from ppo_algorithm.ppo_hyperparams import ppo_params
from ppo_algorithm.normalizer import Normalizer


def run_ppo(env, training_iterations, num_actors, ppo_epochs, trajectory_size, vis=False, plot=False):
    """
    This method runs ppo_algorithm algorithm.

    :param env: gym environment
    :param training_iterations: number of training iterations
    :param ppo_epochs: number of ppo_algorithm epochs
    :param vis: if true than show visualization
    :return:
    """
    # if plot:
    #     plt.axis([0, ppo_params['plot_time_axis'], 0, ppo_params['plot_reward_axis']])
    #     plt.ion()
    #     plt.show()

    env = GentlyTerminating(env)

    # ---------------------------------------- #
    #### initialize actor-critic networks ####
    # ---------------------------------------- #
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    num_hidden_neurons = ppo_params['num_hidden_neurons']

    state_normalizer = Normalizer(num_inputs)
    return_normalizer = Normalizer(1)


    ac_net = actor_critic.ActorCriticMLPShared(num_inputs, num_hidden_neurons,
                                               num_outputs, std=ppo_params['actor_network_std'])
    # print(list(ac_net.parameters()))
    total_rewards = []
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
        advantage_estimates = np.array([])
        returns = np.array([])
        for i in range(num_actors):

            state = env.reset()
            total_reward = 0
            for t in range(trajectory_size):
                state = torch.FloatTensor(state)
                state_normalizer.observe(state)
                state = state_normalizer.normalize(state)

                # sample state from normal distribution
                dist, value = ac_net(state)

                # action = np.clip(dist.sample(), env.action_space.low, env.action_space.high)
                action = dist.sample()

                next_state, reward, done, info = env.step(action.cpu().detach().numpy()[0])
                # save values and rewards for gae
                log_prob = dist.log_prob(action)
                values.append(value)
                old_log_probs.append(log_prob)
                states.append(state)
                actions.append(action)
                state = next_state
                rewards.append(reward)
                masks.append(1 - done)
                total_reward += reward

                if done:
                    state = env.reset()

            _, last_value = ac_net(torch.FloatTensor(next_state))

            # ------------------------------------------------------ #
            #### compute advantage estimates from trajectory ####
            # ------------------------------------------------------ #
            if vis and epoch % 100 == 0:
                cum_r = render_policy(env, ac_net, state_normalizer)
                # print("Cumulative reward: {} in epoch: {}".format(cum_r, epoch))
            advantage_estimates_, returns_ = compute_gae(rewards[i*trajectory_size:i*trajectory_size+trajectory_size],
                                                         values[i*trajectory_size:i*trajectory_size+trajectory_size],
                                                         last_value,
                                                         masks[i*trajectory_size:i*trajectory_size+trajectory_size],
                                                         ppo_params['advantage_discount'],
                                                         ppo_params['bias_var_trade_off'])

            advantage_estimates = np.concatenate((advantage_estimates, advantage_estimates_))
            returns = np.concatenate((returns, returns_))
        values = torch.cat(values).detach()
        old_log_probs = torch.cat(old_log_probs).detach()
        actions = torch.cat(actions)


        if epoch % 1 == 0:
            print('#############################################')
            print('Total reward: {} in epoch: {}'.format(total_reward/ppo_params['num_actors'], epoch))
            print('Variance: {}'.format(dist.variance.detach().numpy().squeeze()))
        total_rewards.append(total_reward/ppo_params['num_actors'])

        ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs,
                   ac_net, ppo_params['minibatch_size'], returns, ppo_params['clipping'])

        if plot and epoch % 200 == 0:
            plt.plot(total_rewards)
            plt.show()
            # plt.draw()
            # plt.pause(0.001)

def ppo_update(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs,
               ac_net, minibatch_size, returns, eps=0.2):
    """
    This method performs proximal policy update over batches of inputs

    :param ppo_epochs: number of ppo optimization epochs per trajectory
    :param advantage_estimates: computed advantage estimates
    :param states: collected number of states of a trajectory
    :param actions: collected number of actions of a trajectory
    :param values: collected number of values of a trajectory
    :param old_log_probs: old log probabilities.
    :param actor_net: current actor network (samples policy)
    :param critic_net: current critic network (sample new values)
    :param minibatch_size: size of minibatches for each ppo epoch
    :return:
    """
    lr = ppo_params['optim_lr']
    ac_optim = optim.Adam(ac_net.parameters(), lr=lr)

    # constants for surrogate objective
    c1,c2 = ppo_params['critic_loss_coeff'], ppo_params['entropy_loss_coeff']

    randomized_inds = np.arange(len(states))

    # shape states for further processing
    states = torch.stack(states)

    # normalize advantage estimates
    # advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / \
    #                       (advantage_estimates.std() + 1e-8)

    for k in range(ppo_epochs):
        # shuffle inputs every ppo epoch
        np.random.shuffle(randomized_inds)
        old_log_probs = old_log_probs[randomized_inds]
        actions = actions[randomized_inds]
        values = values[randomized_inds]
        advantage_estimates = advantage_estimates[randomized_inds]
        states = states[randomized_inds]
        returns = returns[randomized_inds]

        # frac = 1.0 - (k - 1.0) / ppo_epochs
        # Calculate the learning rate
        # lrnow = lr(frac)

        for i in range(0, len(states), minibatch_size):
            dist, current_policy_value = ac_net(states[i:i+minibatch_size])

            new_log_prob = dist.log_prob(actions[i:i+minibatch_size])
            entropy = dist.entropy().mean()


            ratio = torch.exp(new_log_prob - old_log_probs[i:i+minibatch_size])

            # shape advantage estimate vector
            # advantage_batch = torch.FloatTensor(advantage_estimates[i:i+minibatch_size]).view(advantage_estimates[i:i+minibatch_size].shape[0], 1)
            advantage_batch = torch.FloatTensor(advantage_estimates[i:i+minibatch_size])

            surr = ratio * advantage_batch
            clipped_surr = torch.clamp(ratio, 1-eps, 1+eps) * advantage_batch

            actor_loss = torch.min(surr, clipped_surr).mean()
            # critic_loss = ((advantage_batch - values[i:i+minibatch_size] - target_value) ** 2).mean()

            target_value = torch.FloatTensor(returns[i:i + minibatch_size]).detach()
            critic_loss = ((current_policy_value - target_value).pow(2)).mean()

            # loss = -(actor_loss + c1 * critic_loss - c2 * entropy)
            loss = -(actor_loss - c1 * critic_loss + c2 * entropy)

            # loss = actor_loss - c1 * critic_loss + c2 * entropy

            ac_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_net.parameters(), ppo_params['max_grad_norm'])
            ac_optim.step()

def render_policy(env, policy, normalizer=None):
    done = False
    state = env.reset()
    cum_reward = 0
    while not done:
        state = torch.FloatTensor(state)
        state = normalizer.normalize(state)
        dist, _ = policy(torch.FloatTensor(state))
        action = dist.sample()
        state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
        cum_reward += reward
        env.render()
    print('||||||||||||||||||||||||||||||')
    print('Cumulative reward:', cum_reward)
    print('||||||||||||||||||||||||||||||')
    return cum_reward




