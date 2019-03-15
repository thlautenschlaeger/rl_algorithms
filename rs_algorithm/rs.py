import numpy as np
from rs_algorithm.utils import model_handler


class RandomSearch:

    def __init__(self, env, hyperparams, path, resume_training=False):
        self.env = env  # the environment used for training
        self.epochs = hyperparams['num_iterations']  # number of iterations performed by RS
        self.alpha = hyperparams['lr']  # learning rate
        self.num_deltas = hyperparams['num_deltas']  # number of directions explored in RS
        self.sample_noise = hyperparams['sample_noise']  # stretch of exploring directions
        self.bbest = hyperparams['bbest']  # number of directions used for update
        self.horizon = hyperparams['horizon']  # length of horizon
        self.num_features = hyperparams['num_features']  # number of features used for V3
        self.n = self.env.observation_space.shape[0]  # dimension of state space
        self.p = self.env.action_space.shape[0]  # dimension of action space
        self.number_of_encountered_states_pre_update = 0  # used for calculating mean for V2
        self.number_of_encountered_states_post_update = 0  # used for calculating mean for V2
        self.state_mean = np.zeros(shape=self.n)  # used for calculating mean for V2
        self.mean_of_encountered_states_post_update = np.zeros(shape=self.n)  # used for calculating mean for V2
        self.sigma_diag_pre_update = np.ones(self.n)  # used for calculating sigma for V2
        self.sigma_diag_post_update = np.zeros(self.n)  # used for calculating sigma for V2
        self.features = []  # list of features
        self.path = path  # path for saving models
        self.hyperparams = hyperparams  # hyper params
        self.eval_step = hyperparams['eval_step']  # step size for evaluating the learned policy
        self.resume_training = resume_training  # flag for resuming training

    def ars_v1(self):
        """
        Implementation of V1 of Random Search. Please refer to project report or
        original paper for detailed explanation
        :return: all rewards obtained during evaluation of the policy
        """
        collected_eval_rewards = []
        M = np.full((self.p, self.n), 0)
        if self.resume_training:
            M, collected_eval_rewards = model_handler.load_policy_v1(self.path)
        collected_eval_rewards.append(evaluate_policy_v1(self.env, M))
        max_evaluated_policy_reward = np.NINF
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.n))
            total_rewards = np.empty((self.num_deltas, 2))
            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(M, deltas[k])
                total_reward_plus = self.perform_rollouts_v1(self.horizon, mj_plus)
                total_reward_minus = self.perform_rollouts_v1(self.horizon, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)
            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards, std_rewards = self.compute_sum_b_best_rewards(deltas, sorted_ids, total_rewards)
            M = self.update_policy(M, std_rewards, sum_b_best_rewards)

            if epoch % self.eval_step == 0 and epoch != 0:
                evaluated_policy_reward = evaluate_policy_v1(self.env, M)
                collected_eval_rewards.append(evaluated_policy_reward)
                print("Expected reward: {} in {} epochs".format(evaluated_policy_reward, epoch))
                if evaluated_policy_reward > max_evaluated_policy_reward:
                    max_evaluated_policy_reward = evaluated_policy_reward
                    model_handler.save_model_ars_v1(M, collected_eval_rewards, self.path)

        return collected_eval_rewards

    def ars_v1_ff(self):
        """
        Implementation of own version of random search. It is referred as V3 in project report.
        :return: all rewards obtained during evaluation of the policy
        """
        self.init_random_fourier_functions(self.num_features)
        linear_policy = np.full((self.p, self.num_features + 1), 0)
        max_evaluated_policy_reward = np.NINF
        collected_eval_rewards = []
        if self.resume_training:
            linear_policy, self.features, collected_eval_rewards = model_handler.load_policy_v1_rff(self.path)
            self.num_features = len(self.features)
        collected_eval_rewards.append(evaluate_policy_v1_rff(self.env, linear_policy, self.features, render=False))
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.num_features + 1))
            total_rewards = np.empty((self.num_deltas, 2))
            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(linear_policy, deltas[k])
                total_reward_plus = self.perform_rollouts_v1_rff(self.horizon, mj_plus)
                total_reward_minus = self.perform_rollouts_v1_rff(self.horizon, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)
            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards, std_rewards = self.compute_sum_b_best_rewards(deltas, sorted_ids, total_rewards)
            linear_policy = self.update_policy(linear_policy, std_rewards, sum_b_best_rewards)
            if epoch % self.eval_step == 0 and epoch != 0:
                evaluated_policy_reward = evaluate_policy_v1_rff(self.env, linear_policy, self.features, render=False)
                collected_eval_rewards.append(evaluated_policy_reward)
                print("Expected reward: {} in {} epochs".format(evaluated_policy_reward, epoch))
                if evaluated_policy_reward > max_evaluated_policy_reward:
                    max_evaluated_policy_reward = evaluated_policy_reward
                    model_handler.save_model_ars_v1_rff(linear_policy, self.features, collected_eval_rewards, self.path)
        return collected_eval_rewards

    def ars_v2(self):
        """
        Implementation of V2 of Random Search. Please refer to project report or
        original paper for detailed explanation
        :return: all rewards obtained during evaluation of the policy
        """
        M = np.zeros(shape=(self.p, self.n))
        max_evaluated_policy_reward = np.NINF
        collected_eval_rewards = []
        if self.resume_training:
            M, self.sigma_diag_pre_update, self.state_mean, self.number_of_encountered_states_pre_update, \
                collected_eval_rewards = model_handler.load_policy_v2(self.path)
            self.sigma_diag_post_update = self.sigma_diag_pre_update
            self.mean_of_encountered_states_post_update = self.state_mean
            self.number_of_encountered_states_post_update = self.number_of_encountered_states_pre_update
        collected_eval_rewards.append(evaluate_policy_v2(self.env, M, self.compute_sigma_rooted(self.sigma_diag_pre_update),
                                                         self.mean_of_encountered_states_post_update))
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.n))
            total_rewards = np.zeros((self.num_deltas, 2))
            sigma_rooted = self.compute_sigma_rooted()
            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(M, deltas[k])
                total_reward_plus = self.perform_rollouts_v2(self.horizon, mj_plus, self.state_mean, sigma_rooted)
                total_reward_minus= self.perform_rollouts_v2(self.horizon, mj_minus, self.state_mean, sigma_rooted)
                total_rewards[k] = (total_reward_plus, total_reward_minus)

            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards, std_rewards = self.compute_sum_b_best_rewards(deltas, sorted_ids, total_rewards)
            M = self.update_policy(M, std_rewards, sum_b_best_rewards)
            self.sigma_diag_pre_update = self.sigma_diag_post_update
            self.state_mean = self.mean_of_encountered_states_post_update
            evaluated_policy_reward = evaluate_policy_v2(self.env, M, sigma_rooted,
                                                         self.mean_of_encountered_states_post_update,
                                                         self.horizon)
            collected_eval_rewards.append(evaluated_policy_reward)

            if epoch % self.eval_step == 0 and epoch != 0:
                evaluated_policy_reward = evaluate_policy_v2(self.env, M, sigma_rooted,
                                                             self.mean_of_encountered_states_post_update)
                collected_eval_rewards.append(evaluated_policy_reward)
                print("Expected reward: {} in {} epochs".format(evaluated_policy_reward, epoch))
                if evaluated_policy_reward > max_evaluated_policy_reward:
                    max_evaluated_policy_reward = evaluated_policy_reward
                    model_handler.save_model_ars_v2(M, self.sigma_diag_post_update,
                                                    self.mean_of_encountered_states_post_update, collected_eval_rewards,
                                                    self.number_of_encountered_states_post_update, self.path)
        return collected_eval_rewards

    def sample_policy(self, M, delta):
        """
        Computes a new M for evaluation
        :param : Last M
        :param delta: direction in which policy is changed
        :returns: the two modified Ms
        """
        mj_plus = (M + self.sample_noise * delta)
        mj_minus = (M - self.sample_noise * delta)
        return mj_minus, mj_plus

    def compute_sum_b_best_rewards(self, deltas, sorted_ids, total_rewards):
        """
        Computes the sum of the b best rewards of one iteration of RS
        :param deltas: all directions used during the iteration
        :param sorted_ids: ids indicating the rank of a policy
        :param total_rewards: all rewards obtained during a iteration
        :return: the b best rewards and its standard deviation
        """
        sum_b_best_rewards = 0
        b_best_rewards = np.empty((2 * self.bbest))
        for i in range(self.bbest):
            id_x = sorted_ids[i]
            sum_b_best_rewards += (total_rewards[id_x][0] - total_rewards[id_x][1]) * deltas[id_x]
            b_best_rewards[2 * i] = total_rewards[id_x][0]
            b_best_rewards[2 * i + 1] = total_rewards[id_x][1]
        return sum_b_best_rewards, np.std(b_best_rewards)

    def update_policy(self, M, std_rewards, sum_b_best_rewards):
        """
        Policy gets updated
        :param M: old policy M
        :param std_rewards: stddev of rewards
        :param sum_b_best_rewards: sum of b best rewards
        :return: new policy
        """
        M = M + (self.alpha / (self.bbest * std_rewards)) * sum_b_best_rewards
        return M

    def perform_rollouts_v2(self, H, M, mean, sigma_rooted):
        """ runs a policy
        :param H: length of the horizon
        :param M: M as in paper
        :param mean: mean of all encountered states
        :param sigma_rooted: sigma as in paper
        :return: the total reward obtained from the policy
        """
        total_reward = 0
        state = self.env.reset()
        self.update_mean_std(state)
        for i in range(H):
            action = np.matmul(np.matmul(M, sigma_rooted), (state-mean))
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            self.update_mean_std(state)
            total_reward += reward
            if done:
                break
        return total_reward

    def compute_sigma_rooted(self):
        """
        computes sigma rooted necessary for calculating action.
        :return: sigma_rooted
        """
        sigma_rooted = np.identity(self.n)
        for i in range(len(self.sigma_diag_pre_update)):
            if self.sigma_diag_pre_update[i] < 0.00000008:
                sigma_rooted[i][i] = np.power(np.inf,-0.5)
            else:
                sigma_rooted[i][i] = np.power(self.sigma_diag_pre_update[i], -0.5)
        return sigma_rooted

    def update_mean_std(self, state):
        """
        updates the mean of all encountered states during training

        :param state: environment state
        """
        nr_states = self.number_of_encountered_states_post_update
        old_mean =  self.mean_of_encountered_states_post_update
        new_mean = old_mean + (state - old_mean) / (nr_states+1)

        old_sigma_diag = self.sigma_diag_post_update
        new_sigma_diag = np.zeros(len(old_sigma_diag),dtype=np.float128)
        for i in range(len(old_sigma_diag)):
            a = state[i] - old_mean[i]
            b = state[i] - new_mean[i]
            c = a *b
            d = c - old_sigma_diag[i]
            e = d/(nr_states+1)
            new_sigma_diag[i] = e + old_sigma_diag[i]

        self.mean_of_encountered_states_post_update = new_mean
        self.number_of_encountered_states_post_update += 1
        self.sigma_diag_post_update = new_sigma_diag

    def perform_rollouts_v1_rff(self, H, weights):
        """
        runs the policy obtained from weights
        :param H: length of horizon
        :param weights: parameters of policy
        :return the cumulative reward achieved by the policy
        """
        total_reward = 0
        state = self.env.reset()
        for i in range(H):
            state = transform_state(state, self.features)
            action = np.matmul(weights, state)[0]
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward

    def init_random_fourier_functions(self, features_nr):
        """ randomly initializes fourier features
        :param features_nr: number of features to be initialized
        """
        for i in range(features_nr):
            self.features.append(np.random.randint(features_nr + 1, size=self.n))

    def perform_rollouts_v1(self, horizon, weights):
        """ Runs a policy
        :param env: gym environment
        :param horizon: rollout horizon
        :param weights: linear policy
        :return: total rewards
        """
        total_reward = 0
        state = self.env.reset()
        for i in range(horizon):
            action = np.matmul(weights, state[np.newaxis].T)[0]
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward


def sort_max_index_reversed(arr):
    """ Returns an array consisting of sorted indexes
    :param arr: the array to sort
    :return: an array with sorted indexes
    """
    n = len(arr)
    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = max(arr[i])
    return np.argsort(tmp)[::-1]


def evaluate_policy_v2(env, weights, sigma_rooted, mean, render=False):
    """ evaluates a policy
    :param env: gym environment
    :param weights: linear policy
    :param sigma_rooted: covar matrix
    :param mean: the mean of encountered states
    :param render: bool
    :return: total reward
    """
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.matmul(np.matmul(weights, sigma_rooted), (state - mean))
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if render:
            env.render()
    return total_reward


def evaluate_policy_v1_rff(env, weights, features, render=False):
    """
    :param env: gym environment
    :param weights: linear policy
    :param features: fourier features
    :param horizon: rollout horizon
    :param render: bool

    :return: total reward
    """
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = transform_state(state, features)
        action = np.matmul(weights, state)[0]
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward
        if render:
            env.render()

    return total_reward


def evaluate_policy_v1(env, weights, render=False):
    """
    :param env: gym environment
    :param weights: linear policy
    :param horizon: rollout horizon
    :param render: bool
    :return: total reward
    """
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.matmul(weights, state[np.newaxis].T)[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if render:
            env.render()

    return total_reward


def transform_state(state, features):
    """ maps a state to its fourier features
    :param state: environment state
    :param features: the random fourier features used for transformation
    :return the transformed state
    """
    transformed_state = np.zeros(len(features)+1)
    for i in range(len(features)):
        transformed_state[i] = np.cos(np.matmul(state, features[i]))
    transformed_state[len(features)] = 0.5
    return transformed_state[np.newaxis].T

