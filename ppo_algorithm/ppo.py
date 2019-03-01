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

    # def __init__(self, env, num_iterations, num_actors, ppo_epochs,
    #              trajectoriy_size, hidden_neurons, std, batch_size, model=None,
    #              optimizer=None, vis=False, plot=False, continue_training=False):
    #
    #     self.env = GentlyTerminating(env)
    #     self.num_iterations = num_iterations
    #     self.num_actors = num_actors
    #     self.ppo_epochs = ppo_epochs
    #     self.trajectory_size = trajectoriy_size
    #     self.batch_size = batch_size
    #     self.vis = vis
    #     self.plot = plot
    #
    #     self.num_inputs = self.env.observation_space.shape[0]
    #     self.num_outputs = self.env.action_space.shape[0]
    #     self.ac_net = actor_critic.ActorCriticMLPShared__(num_inputs=self.num_inputs,
    #                                                      num_hidden_neurons=hidden_neurons,
    #                                                      num_outputs=self.num_outputs,
    #                                                      std=std)
    #
    #     self.state_normalizer = Normalizer(num_inputs=self.num_inputs)
    #     self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=ppo_params['optim_lr'])
    #
    #     # if continue_training:
    #     #     load_files =

    def __init__(self, env, num_iterations, num_actors, ppo_epochs,
                 trajectoriy_size, hidden_neurons, std, batch_size,
                  vis=False, plot=False):

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
        self.num_states = self.num_inputs
        self.ac_net = actor_critic.ActorCriticMLPShared__(num_inputs=self.num_inputs,
                                                         num_hidden_neurons=hidden_neurons,
                                                         num_outputs=self.num_outputs,
                                                         std=std)

        self.state_normalizer = Normalizer(num_inputs=self.num_inputs)
        self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=ppo_params['optim_lr'])

    def run_trajectory(self):
        rewards = np.empty(shape=self.trajectory_size)
        values = torch.empty(self.trajectory_size)
        states = torch.empty(size=(self.trajectory_size, self.num_states))
        actions = torch.empty(self.trajectory_size, 1)
        masks = np.empty(self.trajectory_size)
        old_log_probs = torch.empty(size=(self.trajectory_size, 1))
        state = self.env.reset()
        cum_reward = 0

        for i in range(self.trajectory_size):
            state = torch.FloatTensor(state)
            # self.state_normalizer.observe(state)
            # state = self.state_normalizer.normalize(state)

            # sample state from normal distribution
            dist, value = self.ac_net(state)

            action = dist.sample()

            next_state, reward, done, info = self.env.step(action.cpu().detach().numpy()[0])

            # save values and rewards for gae
            log_prob = dist.log_prob(action)
            values[i] = value
            old_log_probs[i] = log_prob
            states[i] = state
            actions[i] = action
            state = next_state
            rewards[i] = reward
            masks[i] = 1-done
            cum_reward += reward

            if done:
                state = self.env.reset()

        _, last_value = self.ac_net(torch.FloatTensor(next_state))
        values = values.detach()
        last_value = last_value.detach()
        old_log_probs = old_log_probs.detach()

        print('std: {} variance {}'.format(self.ac_net.log_std.exp().detach().numpy().squeeze(), dist.variance.detach().numpy().squeeze()))
        return values, old_log_probs, actions, states, rewards, last_value, masks

    def ppo_update(self, advantage_estimates, states, actions, values, old_log_probs,
                    returns, ep=0.2):
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

        # constants for surrogate objective
        c1, c2 = ppo_params['critic_loss_coeff'], ppo_params['entropy_loss_coeff']

        randomized_inds = np.arange(self.trajectory_size)

        # shape states for further processing
        # states = torch.stack(states)

        # advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / \
        #                       (advantage_estimates.std() + 1e-8)

        for k in range(self.ppo_epochs):
            # shuffle inputs every ppo epoch
            np.random.shuffle(randomized_inds)
            old_log_probs = old_log_probs[randomized_inds]
            actions = actions[randomized_inds]
            values = values[randomized_inds]
            advantage_estimates = advantage_estimates[randomized_inds]
            states = states[randomized_inds]
            returns = returns[randomized_inds]
            frac = 1.0 - (k) / (self.ppo_epochs + 1)

            self.ac_optim.param_groups[0]['lr'] = ppo_params['optim_lrr'](frac, ppo_params['optim_lr'])
            eps = ep(frac)

            for i in range(0, self.trajectory_size, self.batch_size):
                dist, current_policy_value = self.ac_net(states[i:i + self.batch_size])

                new_log_prob = dist.log_prob(actions[i:i + self.batch_size])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_log_probs[i:i + self.batch_size])

                advantage_batch = torch.FloatTensor(advantage_estimates[i:i + self.batch_size])

                surr = ratio * advantage_batch
                clipped_surr = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage_batch

                pg_loss = torch.min(surr, clipped_surr).mean()

                target_value = torch.FloatTensor(returns[i:i + self.batch_size]).detach()
                vf_loss = ((current_policy_value - target_value).pow(2)).mean()

                loss = -(pg_loss - c1 * vf_loss + c2 * entropy)

                self.ac_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), ppo_params['max_grad_norm'])
                self.ac_optim.step()


    def run_ppo(self):
        folder_num = 2
        check_reward = 0
        cum_rewards = []
        render_rewards = []
        path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)
        for i in range(self.num_iterations):

            values, old_log_probs, actions, states, rewards, last_value, masks = self.run_trajectory()

            advantage_est, returns = compute_gae(rewards, values, last_value, masks,
                                                 ppo_params['advantage_discount'], ppo_params['bias_var_trade_off'])

            total_rewards = rewards.sum()

            self.ppo_update(advantage_est, states, actions, values,
                            old_log_probs, returns, ep=ppo_params['clipping'])
            print("Reward: {} in epoch: {}".format(rewards.sum(), i))
            print("############################################:", folder_num)

            cum_rewards.append(total_rewards)

            if check_reward < total_rewards:
                print("SAVE NEW MODEL")
                check_reward = total_rewards
                save_model.save_files(self.ac_net, self.ac_optim, np.array(cum_rewards), np.array(render_rewards),
                                      self.state_normalizer, path)

            # plotting and evaluating policy
            if i % 100 == 0:
                plt.plot(cum_rewards)
                # plt.show()
                plt.savefig(path+'/reward.png')
                render_reward = render_policy(self.env, self.ac_net, normalizer=self.state_normalizer)
                render_rewards.append(render_reward)

def render_policy(env, policy, normalizer=None):
    done = False
    state = env.reset()
    cum_reward = 0
    while not done:
        state = torch.FloatTensor(state)
        # state = normalizer.normalize(state)
        dist, _ = policy(torch.FloatTensor(state))
        action = dist.sample()
        state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
        cum_reward += reward
        # env.render()
    print('||||||||||||||||||||||||||||||')
    print('Cumulative reward:', cum_reward)
    print('||||||||||||||||||||||||||||||')
    return cum_reward