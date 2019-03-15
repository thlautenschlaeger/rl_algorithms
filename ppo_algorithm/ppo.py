from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
import numpy as np
from ppo_algorithm.gae import compute_gae
from ppo_algorithm.models import actor_critic
from ppo_algorithm.utils import plot_utility
from ppo_algorithm.utils import model_handler
from torch.distributions import Normal

class PPO():

    def __init__(self, env, path, hyper_params, continue_training=False):
        """
        This class provides a PPO implementation

        :param env: gym environment
        :param path: path where to save checkpoints and results
        :param hyper_params: hyper parameter for ppo
        :param continue_training: checks if to continue training or start new
        """

        self.env = GentlyTerminating(env)
        self.path = path
        self.num_iterations = hyper_params['num_iterations']    # number of total training iterations
        self.lamb = hyper_params['lambda']                      # lambda for general advantage estimate
        self.cliprange  =hyper_params['cliprange']              # ppo cliprange of importance weights
        self.gamma = hyper_params['gamma']                      # gamma for general advantage estimate
        self.ppo_epochs = hyper_params['ppo_epochs']            # number ppo optimization epochs
        self.horizon = hyper_params['horizon']                  # number of training samples per iteration
        self.minibatches = hyper_params['minibatches']          # minibatch size for ppo optimization
        self.vf_coef = hyper_params['vf_coef']                  # value function coefficient
        self.entropy_coef = hyper_params['entropy_coef']        # entropy coefficient
        self.num_hidden_neurons = hyper_params['num_hidden_neurons']    # number of hidden neurons
        self.policy_std = hyper_params['policy_std']            # initial policy stddev
        self.lr = hyper_params['lr']                            # lern rate
        self.max_grad_norm = hyper_params['max_grad_norm']      # maximum gradient norm for param update
        self.num_evals = hyper_params['num_evals']              # number of policy evaluations to compute expected reward
        self.eval_step = hyper_params['eval_step']              # policy gets evaluated after every eval_step

        self.num_inputs = self.env.observation_space.shape[0]
        self.num_outputs = self.env.action_space.shape[0]
        self.num_states = self.num_inputs
        self.cumulative_rollout_rewards = np.array([])
        self.cum_eval_rewards = np.array([])
        self.cum_eval_rewards_std = np.array([])
        self.entropy = np.array([])
        self.epoch = 0

        # initialize actor critic network
        self.ac_net = actor_critic.ActorCriticMLPShared(num_inputs=self.num_inputs,
                                                        num_hidden_neurons=self.num_hidden_neurons,
                                                        num_outputs=self.num_outputs,
                                                        layer_norm=hyper_params['layer_norm'],
                                                        std=self.policy_std)


        self.ac_optim = optim.Adam(self.ac_net.parameters(), lr=self.lr)

        if continue_training:
            self.ac_net, self.ac_optim, self.cumulative_rollout_rewards, \
            self.cum_eval_rewards, self.cum_eval_rewards_std, self.epoch, self.entropy = \
                model_handler.load_model(path=path,
                                         model=self.ac_net,
                                         optimizer=self.ac_optim,
                                         from_checkpoint=True)
            self.ac_net.train()


    def collect_trajectories(self):
        """
        collects multiple trajectories limited by horizon

        :return: values,old_log_probs, actions, states, rewards, masks, entropy
        """
        # init arrays for data collection
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
        last_value = last_value.detach()
        values = values.detach()
        entropy = dist.entropy().detach().numpy()[0][0]
        old_log_probs = old_log_probs.detach()

        return values, old_log_probs, actions, states, rewards, last_value, masks, entropy

    def ppo_update(self, advantage_estimates, states, actions, old_log_probs,
                   returns, cliprange=0.2):
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

        """
        randomized_inds = np.arange(self.horizon)

        # normalize advantages
        advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / \
                              (advantage_estimates.std() + 1e-8)
        for k in range(self.ppo_epochs):
            # shuffle inputs every ppo epoch
            np.random.shuffle(randomized_inds)
            old_log_probs = old_log_probs[randomized_inds]
            actions = actions[randomized_inds]
            advantage_estimates = advantage_estimates[randomized_inds]
            states = states[randomized_inds]
            returns = returns[randomized_inds]

            for start in range(0, self.horizon, self.minibatches):
                end = start + self.minibatches
                mean, std, current_policy_value = self.ac_net(states[start:end])

                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(actions[start:end])
                entropy = dist.entropy().mean()

                # importance weights
                ratio = torch.exp(new_log_prob - old_log_probs[start:end])

                advantage_batch = advantage_estimates[start:end]

                surr = ratio * advantage_batch
                clipped_surr = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantage_batch
                pg_loss = torch.min(surr, clipped_surr).mean()

                target_value = returns[start:end]
                vf_loss = ((current_policy_value - target_value).pow(2)).mean()

                loss = -(pg_loss - self.vf_coef * vf_loss + self.entropy_coef * entropy)

                self.ac_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), self.max_grad_norm)
                self.ac_optim.step()

    def run_ppo(self):
        """
        runs ppo and logs data

        """
        check_reward = 0
        for epoch in range(self.epoch, self.num_iterations + 1):

            # collect trajectory data
            values, old_log_probs, actions, states, \
            rewards, last_value, masks, entropy = self.collect_trajectories()

            # computes general advantages from trajectories
            advantage_est, returns = compute_gae(rewards, values, last_value, masks, self.lamb, self.gamma)

            # interesting to check how model behaves
            total_rollout_reward = rewards.sum()

            self.cumulative_rollout_rewards = np.append(self.cumulative_rollout_rewards, total_rollout_reward)
            self.entropy = np.append(self.entropy, entropy)

            # plotting and evaluating policy
            if epoch % self.eval_step == 0:
                check_reward = self.logger(check_reward, epoch)

            # actual ppo optimization
            self.ppo_update(advantage_est, states, actions,
                            old_log_probs, returns, cliprange=self.cliprange)

    def logger(self, check_reward, epoch):
        """
        evaluates current model, checks if to save best scoring model.

        :param check_reward:
        :param epoch:
        :return:
        """
        eval_reward, eval_std = eval_policy(env=self.env,
                                            model=self.ac_net,
                                            num_evals=self.num_evals, )
        self.cum_eval_rewards = np.append(self.cum_eval_rewards, eval_reward)
        self.cum_eval_rewards_std = np.append(self.cum_eval_rewards_std, eval_std)
        plot_utility.plt_expected_cum_reward(self.path, self.cum_eval_rewards, self.eval_step)
        print("---------------------------------------------------")
        print("Expected cumulative reward: {} after {} epochs:".format(eval_reward, epoch))
        print("---------------------------------------------------")
        model_handler.save_model(model=self.ac_net,
                                 optimizer=self.ac_optim,
                                 train_rewards=self.cumulative_rollout_rewards,
                                 eval_rewards=self.cum_eval_rewards,
                                 eval_rewards_std=self.cum_eval_rewards_std,
                                 epoch=epoch,
                                 entropy=self.entropy,
                                 path=self.path + '/checkpoint')
        if check_reward < eval_reward:
            print("Found new high scoring model")
            check_reward = eval_reward
            model_handler.save_model(model=self.ac_net,
                                     optimizer=self.ac_optim,
                                     train_rewards=self.cumulative_rollout_rewards,
                                     eval_rewards=self.cum_eval_rewards,
                                     eval_rewards_std=self.cum_eval_rewards_std,
                                     epoch=epoch,
                                     entropy=self.entropy,
                                     path=self.path + '/best_policy')

        return check_reward


def eval_policy(env, model, num_evals=0):
    """
    evaluates given model

    :param env: gym environment
    :param model: actor-critic model
    :param num_evals: number of model evaluations

    :return: expected model reward and corresponding stddevs
    """
    cum_reward_list = np.empty(num_evals)
    for i in range(num_evals):
        cum_reward = 0
        done = False
        state = env.reset()
        while not done:
            state = torch.FloatTensor(state)
            mean, std, _ = model(state)
            dist = Normal(mean, 0)
            action = dist.sample()
            state, reward, done, _ = env.step(action.cpu().detach().numpy())
            cum_reward += reward
        cum_reward_list[i] = cum_reward
    expected_cum_reward = cum_reward_list.mean()
    expected_cum_reward_std = cum_reward_list.std()
    return expected_cum_reward, expected_cum_reward_std
