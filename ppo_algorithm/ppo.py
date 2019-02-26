from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from ppo_algorithm.gae import compute_gae
from ppo_algorithm.models import actor_critic
from ppo_algorithm.ppo_hyperparams import ppo_params
from ppo_algorithm.normalizer import Normalizer
import time
from ppo_algorithm.utilities import save_model

class PPO():

    def __init__(self, env, num_iterations, num_actors, ppo_epochs,
                 trajectoriy_size, hidden_neurons, std, batch_size, vis=False, plot=False):

        self.env = GentlyTerminating(env)
        self.num_iterations = num_iterations
        self.num_actors = num_actors
        self.ppo_epochs = ppo_epochs
        self.trajectory_size = trajectoriy_size
        self.batch_size = batch_size
        self.vis = vis
        self.plot = plot

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.shape[0]
        self.ac_net = actor_critic.ActorCriticMLPShared_(num_inputs=self.num_inputs,
                                                         num_hidden_neurons=hidden_neurons,
                                                         num_outputs=self.num_outputs,
                                                         std=std)

        self.state_normalizer = Normalizer(num_inputs=self.num_inputs)
        self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=ppo_params['optim_lr'])


    def run_trajectory(self):
        rewards = []
        values = []
        states = []
        actions = []
        masks = []
        old_log_probs = []
        state = self.env.reset()
        cum_reward = 0

        for i in range(self.trajectory_size):
            state = torch.FloatTensor(state)
            #self.state_normalizer.observe(state)
            #state = self.state_normalizer.normalize(state)

            # sample state from normal distribution
            dist, value = self.ac_net(state)
            action = dist.sample()

            next_state, reward, done, info = self.env.step(action.cpu().detach().numpy()[0])
            # next_state, reward, done, info = self.env.step(action.cpu().detach().numpy())
            # save values and rewards for gae
            log_prob = dist.log_prob(action)
            values.append(value)
            old_log_probs.append(log_prob)
            states.append(state)
            actions.append(action)
            state = next_state
            rewards.append(reward)
            masks.append(1 - done)
            cum_reward += reward

            if done:
                state = self.env.reset()

        _, last_value = self.ac_net(torch.FloatTensor(next_state))
        values = torch.cat(values).detach()
        old_log_probs = torch.cat(old_log_probs).detach()
        actions = torch.cat(actions)

        print('Variance: {}'.format(dist.variance.detach().numpy().squeeze()))
        return values, old_log_probs, actions, states, rewards, last_value, masks

    def ppo_update(self, advantage_estimates, states, actions, values, old_log_probs,
                    returns, eps=0.2):
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
        # lr = ppo_params['optim_lr']
        # ac_optim = optim.Adam(self.ac_net.parameters(), lr=lr)

        # constants for surrogate objective
        c1, c2 = ppo_params['critic_loss_coeff'], ppo_params['entropy_loss_coeff']

        randomized_inds = np.arange(len(states))

        # shape states for further processing
        states = torch.stack(states)

        for k in range(self.ppo_epochs):
            # shuffle inputs every ppo epoch
            np.random.shuffle(randomized_inds)
            old_log_probs = old_log_probs[randomized_inds]
            actions = actions[randomized_inds]
            values = values[randomized_inds]
            advantage_estimates = advantage_estimates[randomized_inds]
            states = states[randomized_inds]
            returns = returns[randomized_inds]

            for i in range(0, len(states), self.batch_size):
                dist, current_policy_value = self.ac_net(states[i:i + self.batch_size])

                new_log_prob = dist.log_prob(actions[i:i + self.batch_size])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_log_probs[i:i + self.batch_size])

                advantage_batch = torch.FloatTensor(advantage_estimates[i:i + self.batch_size])

                surr = ratio * advantage_batch
                clipped_surr = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage_batch

                actor_loss = torch.min(surr, clipped_surr).mean()

                target_value = torch.FloatTensor(returns[i:i + self.batch_size]).detach()
                critic_loss = ((current_policy_value - target_value).pow(2)).mean()

                loss = -(actor_loss - c1 * critic_loss + c2 * entropy)

                self.ac_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), ppo_params['max_grad_norm'])
                self.ac_optim.step()


    def run_ppo(self):
        folder_num = 1
        check_reward = 0
        cum_rewards = []
        render_rewards = []
        path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)
        for i in range(self.num_iterations):

            values, old_log_probs, actions, states, rewards, last_value, masks = self.run_trajectory()

            advantage_est, returns = compute_gae(rewards, values, last_value, masks,
                                                 ppo_params['advantage_discount'], ppo_params['bias_var_trade_off'])

            self.ppo_update(advantage_est, states, actions, values,
                            old_log_probs, returns, eps=ppo_params['clipping'])
            print("Reward: {} in epoch: {}".format(np.array(rewards).sum(), i))
            print("############################################:", folder_num)

            cum_rewards.append(np.array(rewards).sum())
            if i % 100 == 0:
                plt.plot(cum_rewards)
                plt.show()
                render_reward = render_policy(self.env, self.ac_net, normalizer=self.state_normalizer)
                render_rewards.append(render_reward)
                if check_reward < render_reward:
                    check_reward = render_reward
                    save_model.save_files(self.ac_net, self.ac_optim, np.array(cum_rewards), np.array(render_rewards), self.state_normalizer, path)
                    # torch.save(self.ac_net, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)+'/ppo_network.pt')
                    # torch.save(self.state_normalizer, '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)+'/normalizer.pt')
                    # np.save('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)+'/rewards.npy', np.array(cum_rewards))
                    # np.save(
                    #     '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/' + str(
                    #         folder_num) + '/render_rewards.npy', np.array(render_rewards))








def run_ppo_old(env, training_iterations, num_actors, ppo_epochs, trajectory_size, vis=False, plot=False):
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
    num_hidden_neurons = ppo_params['num_hidden_neurons']

    state_normalizer = Normalizer(num_inputs)
    return_normalizer = Normalizer(1)


    # ac_net = actor_critic.ActorCriticMLPShared(num_inputs, num_hidden_neurons,
    #                                            num_outputs, std=ppo_params['actor_network_std'])
    ac_net = actor_critic.ActorCriticMLPShared_(num_inputs, [64, 64], num_outputs, std=ppo_params['actor_network_std'])
    total_rewards = []

    for epoch in range(training_iterations):
        rewards = []
        values = []
        states = []
        actions = []
        lel = []
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
                start_time = time.time()
                state = torch.FloatTensor(state)
                lel.append(time.time() - start_time)
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
            print(np.array(lel).mean())
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

        ppo_update_old(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs,
                       ac_net, ppo_params['minibatch_size'], returns, ppo_params['clipping'])

        if plot and epoch % 200 == 0:
            plt.plot(total_rewards)
            plt.show()
            # plt.draw()
            # plt.pause(0.001)

def ppo_update_old(ppo_epochs, advantage_estimates, states, actions, values, old_log_probs,
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
        #state = normalizer.normalize(state)
        dist, _ = policy(torch.FloatTensor(state))
        action = dist.sample()
        state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
        cum_reward += reward
        env.render()
    print('||||||||||||||||||||||||||||||')
    print('Cumulative reward:', cum_reward)
    print('||||||||||||||||||||||||||||||')
    return cum_reward




