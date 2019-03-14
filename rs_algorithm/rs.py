import numpy as np
from rs_algorithm.utils import model_handler


class RandomSearch:

    def __init__(self, env, hyperparams, path, resume_training=False):
        self.env = env
        self.epochs = hyperparams['num_iterations']
        self.alpha = hyperparams['lr']
        self.num_deltas = hyperparams['num_deltas']
        self.sample_noise = hyperparams['sample_noise']
        self.bbest = hyperparams['bbest']
        self.horizon = hyperparams['horizon']
        self.num_features = hyperparams['num_features']
        self.n = self.env.observation_space.shape[0]
        self.p = self.env.action_space.shape[0]
        self.termination_criterion = hyperparams['termination_criterion']
        self.encountered_states = []
        self.number_of_encountered_states_pre_update = 0
        self.number_of_encountered_states_post_update = 0
        self.state_mean = np.zeros(shape=self.n)
        self.mean_of_encountered_states_post_update = np.zeros(shape=self.n)

        self.sigma_diag_pre_update = np.ones(self.n)
        self.sigma_diag_post_update = np.zeros(self.n)
        self.features = []
        self.best_m = 0
        self.path = path
        self.hyperparams = hyperparams
        self.eval_step = hyperparams['eval_step']

        # if resume_training:

    def ars_v1(self):
        """
        Implementation of Augmented Random Search

        """
        all_x_rewards = []
        all_rewards = []
        M = np.full((self.p, self.n), 0)
        max_evaluated_policy_reward = np.NINF
        all_x_rewards.append(evaluate_policy_v1(self.env, M))

        for epoch in range(self.epochs):

            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.n))
            total_rewards = np.empty((self.num_deltas, 2))
            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(M, deltas[k])
                total_reward_plus = self.perform_rollouts_v1(self.horizon, mj_plus)
                total_reward_minus = self.perform_rollouts_v1(self.horizon, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)

            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards, std_rewards = self.evaluate_b_best_rewards(deltas, sorted_ids,
                                                                           total_rewards)

            M = self.update_policy(M, std_rewards, sum_b_best_rewards)
            evaluated_policy_reward = evaluate_policy_v1(self.env, M)
            print('evaluated_reward', evaluated_policy_reward)
            all_rewards.append(evaluated_policy_reward)

            if evaluated_policy_reward > max_evaluated_policy_reward:
                model_handler.save_model_ars_v1(M, all_rewards, self.path + '/best_policy')
                max_evaluated_policy_reward = evaluated_policy_reward

            if epoch % self.eval_step == 0 and epoch != 0:
                print(epoch)
                all_x_rewards.append(evaluate_policy_v1(self.env, M))
                print(all_x_rewards)
                print('----')
        return all_x_rewards


    def ars_v1_rff(self):
        """

        """
        self.init_random_fourier_functions(self.num_features)
        print(self.features)
        linear_policy = np.full((self.p, self.num_features + 1), 0)
        max_evaluated_policy_reward = np.NINF
        collected_eval_rewards = []
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.num_features + 1))
            total_rewards = np.empty((self.num_deltas, 2))

            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(linear_policy, deltas[k])
                total_reward_plus = self.perform_rollouts_v1_rff(self.horizon, mj_plus)
                total_reward_minus = self.perform_rollouts_v1_rff(self.horizon, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)

            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards, std_rewards = self.evaluate_b_best_rewards(deltas, sorted_ids, total_rewards)
            linear_policy = self.update_policy(linear_policy, std_rewards, sum_b_best_rewards)
            eval_reward = evaluate_policy_v1_rff(self.horizon, self.env, linear_policy, self.features)

            collected_eval_rewards.append(eval_reward)
            if eval_reward > max_evaluated_policy_reward:
                model_handler.save_model_ars_v1_rff(linear_policy, self.features, collected_eval_rewards)
                max_evaluated_policy_reward = eval_reward


    def ars_v2(self):

        all_x_rewards = []
        M = np.zeros(shape=(self.p, self.n))
        max_evaluated_policy_reward = np.NINF
        collected_eval_rewards = []
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.num_deltas, self.p, self.n))
            total_rewards = np.zeros((self.num_deltas, 2))
            sigma_rooted = self.compute_sigma_rooted()

            for k in range(self.num_deltas):
                mj_minus, mj_plus = self.sample_policy(M, deltas[k])

                total_reward_plus = self.perform_rollouts_v2(self.horizon, mj_plus, self.state_mean, sigma_rooted)
                total_reward_minus= self.perform_rollouts_v2(self.horizon,mj_minus,self.state_mean,sigma_rooted)
                total_rewards[k] = (total_reward_plus, total_reward_minus)

            # sort reward indices
            sorted_ids = sort_max_index_reversed(total_rewards)

            sum_b_best_rewards, std_rewards = self.evaluate_b_best_rewards(deltas,sorted_ids,total_rewards)

            M = self.update_policy(M, std_rewards, sum_b_best_rewards)

            self.sigma_diag_pre_update = self.sigma_diag_post_update
            self.state_mean = self.mean_of_encountered_states_post_update
            evaluated_policy_reward = evaluate_policy_v2(self.env, M, sigma_rooted,
                                                         self.mean_of_encountered_states_post_update,
                                                         self.horizon)
            collected_eval_rewards.append(evaluated_policy_reward)
            if evaluated_policy_reward > max_evaluated_policy_reward:
                model_handler.save_model_ars_v2(M, sigma_rooted, all_x_rewards, self.path)
                max_evaluated_policy_reward = evaluated_policy_reward


            if epoch % self.eval_step == 0:

                all_x_rewards.append(evaluate_policy_v2(self.env, M, sigma_rooted, self.mean_of_encountered_states_post_update, 1000000))
                print("Save checkpoint")
                print(all_x_rewards)
                print('----')
            print(evaluated_policy_reward)
            print('----------')
        return all_x_rewards


    def sample_policy(self, M, delta):
        mj_plus = (M + self.sample_noise * delta)
        mj_minus = (M - self.sample_noise * delta)
        return mj_minus, mj_plus

    def evaluate_b_best_rewards(self, deltas, sorted_ids, total_rewards):

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
        :return: new policy M

        """

        M = M + (self.alpha / (self.bbest * std_rewards) + 1e-10) * sum_b_best_rewards
        return M

    def perform_rollouts_v2(self, H, M, mean, sigma_rooted):
        """

        :param H:
        :param M:
        :param mean:
        :param sigma_rooted:
        :return:
        """
        total_reward = 0
        action_reward_sequence = []
        state = self.env.reset()
        self.update_mean_std(state)
        for i in range(H):
            action = np.matmul(np.matmul(M, sigma_rooted), (state-mean))
            next_state, reward, done, _ = self.env.step(action)
            action_reward_sequence.append([action, reward])
            state = next_state
            self.update_mean_std(state)
            total_reward += reward
            if done:
                break
        return total_reward, action_reward_sequence

    def compute_sigma_rooted(self):
        sigma_rooted = np.identity(self.n)
        for i in range(len(self.sigma_diag_pre_update)):
            if self.sigma_diag_pre_update[i] < 0.00000008:
                sigma_rooted[i][i] = np.power(np.inf,-0.5)
            else: sigma_rooted[i][i] = np.power(self.sigma_diag_pre_update[i], -0.5)
        return sigma_rooted

    def update_mean_std(self, state):
        """
        updates state mean

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
        total_reward = 0
        state = self.env.reset()
        for i in range(H):
            state = self.transform_state(state)
            action = np.matmul(weights, state)[0]
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward

    def transform_state(self, state):
        transformed_state = np.zeros(len(self.features) + 1)
        for i in range(len(self.features)):
            transformed_state[i] = np.cos(np.matmul(state, self.features[i]))
        transformed_state[len(self.features)] = 0.5
        return transformed_state[np.newaxis].T

    def init_random_fourier_functions(self, features_nr):
        for i in range(features_nr):
            self.features.append(np.random.randint(features_nr + 1, size=self.n))


    def perform_rollouts_v1(self, horizon, weights):
        """

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
    n = len(arr)
    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = max(arr[i])
    return np.argsort(tmp)[::-1]


def evaluate_policy_v2(env, weights, sigma_rooted, mean, render=False):
    """

    :param env: gym environment
    :param weights: linear policy
    :param sigma_rooted: covar matrix
    :param mean:
    :param horizon:
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
        transformed_state = np.zeros(len(features)+1)
        for i in range(len(features)):
            transformed_state[i] = np.cos(np.matmul(state, features[i]))
        transformed_state[len(features)] = 0.5
        return transformed_state[np.newaxis].T



# def save_policy_v1_rff(path, linear_policy, list_of_features, all_rewards)
#     np.save(path + '/linear_policy.npy', linear_policy)
#     np.save(path + '/features.npy', list_of_features)
#     np.save(path + '/training_rewards.npy', all_rewards)
#     # np.save(path + '/hyper_params.npy', hyperparams)
#     # max_evaluated_policy_reward = evaluated_policy_reward