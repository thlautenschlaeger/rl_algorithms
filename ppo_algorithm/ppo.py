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
from ppo_algorithm.utilities import model_handler
from torch.distributions import Normal

class PPO():

    def __init__(self, env, path, hyper_params, continue_training=False):

        self.env = GentlyTerminating(env)
        self.path = path
        self.num_iterations = hyper_params['num_iterations']
        self.lamb = hyper_params['lambda']
        self.cliprange  =hyper_params['cliprange']
        self.gamma = hyper_params['gamma']
        self.ppo_epochs = hyper_params['ppo_epochs']
        self.horizon = hyper_params['horizon']
        self.minibatches = hyper_params['minibatches']
        self.vf_coef = hyper_params['vf_coef']
        self.entropy_coef = hyper_params['entropy_coef']
        self.num_hidden_neurons = hyper_params['num_hidden_neurons']
        self.policy_std = hyper_params['policy_std']
        self.lr = hyper_params['lr']
        self.vf_coef = hyper_params['vf_coef']
        self.max_grad_norm = hyper_params['max_grad_norm']

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.shape[0]
        self.num_states = self.num_inputs
        self.cum_train_rewards = np.array([])
        self.cum_eval_rewards = np.array([])

        self.ac_net = actor_critic.ActorCriticMLPShared(num_inputs=self.num_inputs,
                                                          num_hidden_neurons=self.num_hidden_neurons,
                                                          num_outputs=self.num_outputs,
                                                          std=self.policy_std)

        self.state_normalizer = Normalizer(num_inputs=self.num_inputs)
        self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=self.lr)
        self.decay = lambda f, x: f * x

        if continue_training:
            self.ac_net, self.ac_optim, self.cum_train_rewards, self.cum_eval_rewards = \
                model_handler.load_model(path=path, model=self.ac_net,
                                         optimizer=self.ac_optim, from_checkpoint=True)

    # def __init__(self, env, num_iterations, num_actors, ppo_epochs,
    #              trajectoriy_size, hidden_neurons, policy_std, minibatches,
    #              vis=False, plot=False):
    #
    #     self.env = env
    #     self.num_iterations = num_iterations
    #     self.num_actors = num_actors
    #     self.ppo_epochs = ppo_epochs
    #     self.trajectory_size = trajectoriy_size
    #     self.batch_size = minibatches
    #     self.vis = vis
    #     self.plot = plot
    #
    #     self.num_inputs = self.env.observation_space.shape[0]
    #     self.num_outputs = self.env.action_space.shape[0]
    #     self.num_states = self.num_inputs
    #     self.ac_net = actor_critic.ActorCriticMLPShared(num_inputs=self.num_inputs,
    #                                                     num_hidden_neurons=hidden_neurons,
    #                                                     num_outputs=self.num_outputs,
    #                                                     std=policy_std)

        # self.ac_net2 = actor_critic.ActorCriticMLPShared(num_inputs=self.num_inputs,
        #                                                 num_hidden_neurons=hidden_neurons,
        #                                                 num_outputs=self.num_outputs,
        #                                                 std=policy_std)

        # self.state_normalizer = Normalizer(num_inputs=self.num_inputs)
        # self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=ppo_params['optim_lr'])
        # self.ac_optim2 = optim.Adam(self.ac_net2.parameters(), lr=ppo_params['optim_lr'])
        # self.nets = [self.ac_net, self.ac_net2]
        # self.optims = [self.ac_optim, self.ac_optim2]

    def run_trajectory(self):
        rewards = np.empty(shape=self.horizon)
        values = torch.empty(self.horizon)
        states = torch.empty(size=(self.horizon, self.num_states))
        actions = torch.empty(self.horizon, 1)
        masks = np.empty(self.horizon)
        old_log_probs = torch.empty(size=(self.horizon, 1))
        state = self.env.reset()
        cum_reward = 0

        for i in range(self.horizon):
            state = torch.FloatTensor(state)
            # self.state_normalizer.observe(state)
            # state = self.state_normalizer.normalize(state)

            # sample state from normal distribution
            mean, std, value = self.ac_net(state)

            dist = Normal(mean, std)

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


        _, _, last_value = self.ac_net(torch.FloatTensor(next_state))
        values = values.detach()
        last_value = last_value.detach()
        old_log_probs = old_log_probs.detach()

        print('std: {} variance {}'.format(self.ac_net.log_std.exp().detach().numpy().squeeze(), dist.variance.detach().numpy().squeeze()))
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
        randomized_inds = np.arange(self.horizon)
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

            # self.ac_optim.param_groups[0]['lr'] = ppo_params['optim_lrr'](frac, ppo_params['optim_lr'])
            self.ac_optim.param_groups[0]['lr'] = self.decay(frac, self.lr)
            eps = self.decay(frac, eps)

            for start in range(0, self.horizon, self.minibatches):
                end = start + self.minibatches
                mean, std, current_policy_value = self.ac_net(states[start:end])

                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(actions[start:end])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_prob - old_log_probs[start:end])

                advantage_batch = torch.FloatTensor(advantage_estimates[start:end])

                surr = ratio * advantage_batch
                clipped_surr = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage_batch

                pg_loss = torch.min(surr, clipped_surr).mean()

                target_value = torch.FloatTensor(returns[start:end]).detach()
                vf_loss = ((current_policy_value - target_value).pow(2)).mean()

                loss = -(pg_loss - self.vf_coef * vf_loss + self.entropy_coef * entropy)

                self.ac_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.ac_optim.step()

    def run_ppo(self):
        check_reward = 0
        # path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/'+str(folder_num)
        for i in range(self.num_iterations):

            values, old_log_probs, actions, states, rewards, last_value, masks = self.run_trajectory()

            advantage_est, returns = compute_gae(rewards, values, last_value, masks,
                                                 self.lamb, self.gamma)

            total_rewards = rewards.sum()

            self.ppo_update(advantage_est, states, actions, values,
                            old_log_probs, returns, eps=self.cliprange)
            print("Reward: {} in epoch: {}".format(rewards.sum(), i))
            print("############################################:")

            # cum_rewards.append(total_rewards)
            # np.insert(cum_rewards, -1, total_rewards)
            self.cum_train_rewards = np.append(self.cum_train_rewards, total_rewards)

            if check_reward < total_rewards:
                print("SAVE NEW MODEL")
                check_reward = total_rewards
                model_handler.save_model(self.ac_net, self.ac_optim, self.cum_train_rewards,
                                         self.cum_train_rewards, self.state_normalizer,
                                         self.path +'/best_policy')

            # plotting and evaluating policy
            if i % 100 == 0:
                plt.plot(self.cum_train_rewards)
                # plt.show()
                plt.savefig(self.path+'/reward.png')
                model_handler.save_model(self.ac_net, self.ac_optim, self.cum_train_rewards,
                                         self.cum_train_rewards,
                                         self.state_normalizer, path=self.path+'/checkpoint')
                render_reward = render_policy(self.env, self.ac_net, normalizer=self.state_normalizer)
                self.cum_eval_rewards = np.append(self.cum_eval_rewards, render_reward)


def checkpoint():
    return


def render_policy(env, policy, normalizer=None):
    done = False
    state = env.reset()
    cum_reward = 0
    while not done:
        state = torch.FloatTensor(state)
        # state = normalizer.normalize(state)
        mean, std, _ = policy(torch.FloatTensor(state))
        dist = Normal(mean, std*0)
        action = dist.sample()
        state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
        cum_reward += reward
        env.render()
    print('||||||||||||||||||||||||||||||')
    print('Cumulative reward:', cum_reward)
    print('||||||||||||||||||||||||||||||')
    return cum_reward