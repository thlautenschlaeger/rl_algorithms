import numpy as np
import rs_algorithm.basis_functions as basis_functions


class RandomSearch:

    def __init__(self, env, hyperparams, path):
        self.env = env
        self.epochs = hyperparams['epochs']
        self.alpha = hyperparams['alpha']
        self.N = hyperparams['N']
        self.v = hyperparams['v']
        self.b = hyperparams['b']
        self.H = hyperparams['H']
        self.nr_features = hyperparams['features_nr']
        self.n = self.env.observation_space.shape[0]
        self.p = self.env.action_space.shape[0]
        self.termination_criterion = hyperparams['termination_criterion']
        self.encountered_states = []
        self.number_of_encountered_states_pre_update = 0
        self.number_of_encountered_states_post_update = 0
        self.mean_of_encountered_states_pre_update = np.zeros(shape=self.n)
        self.mean_of_encountered_states_post_update = np.zeros(shape=self.n)
        # self.sigma_diag_pre_update = np.full(self.n, 0.00001)
        # self.sigma_diag_post_update = np.full(self.n, 0.00001)
        self.sigma_diag_pre_update = np.ones(self.n)
        self.sigma_diag_post_update = np.zeros(self.n)
        self.list_of_features = []
        self.best_M = 0
        self.path = path
        self.hyperparams = hyperparams
        self.eval_step = hyperparams['eval_step']


    def ars_v1(self):
        all_x_rewards=[]
        all_rewards = []
        M = np.full((self.p, self.n), 0)
        max_evaluated_policy_reward = np.NINF
        all_x_rewards.append(evaluate_policy_v1(10000000, self.env, M))
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.N, self.p, self.n))
            total_rewards = np.empty((self.N, 2))
            for k in range(self.N):
                mj_plus = (M + self.v * deltas[k])
                mj_minus = (M - self.v * deltas[k])
                total_reward_plus, v1_plus_sequence = perform_rollouts_v1(self.H, self.env, mj_plus)
                total_reward_minus, v1_minus_sequence = perform_rollouts_v1(self.H, self.env, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)
            sorted_ids = sort_max_index_reversed(total_rewards)
            sum_b_best_rewards = 0
            b_best_rewards = np.empty((2 * self.b))
            for i in range(self.b):
                id_x = sorted_ids[i]
                sum_b_best_rewards += (total_rewards[id_x][0] - total_rewards[id_x][1]) * deltas[id_x]
                b_best_rewards[2 * i] = total_rewards[id_x][0]
                b_best_rewards[2 * i + 1] = total_rewards[id_x][1]

            std_rewards = np.std(b_best_rewards)
            M = M + (self.alpha / (self.b * std_rewards)+1e-12) * sum_b_best_rewards
            evaluated_policy_reward = evaluate_policy_v1(self.H, self.env, M)
            print('evaluated_reward', evaluated_policy_reward)
            all_rewards.append(evaluated_policy_reward)
            if evaluated_policy_reward > max_evaluated_policy_reward:
                np.save(self.path + '/M.npy', M)
                np.save(self.path + '/training_rewards.npy', all_rewards)
                np.save(self.path + '/hyper_params.npy', self.hyperparams)
                max_evaluated_policy_reward = evaluated_policy_reward

            if epoch % self.eval_step == 0 and epoch != 0:
                print(epoch)
                all_x_rewards.append(evaluate_policy_v1(10000000, self.env, M))
                print(all_x_rewards)
                print('----')
        return all_x_rewards


    def ars_v1_rff(self):
        self.init_random_fourier_functions(self.nr_features)
        print(self.list_of_features)
        linear_policy = np.full((self.p, self.nr_features+1), 0)
        max_evaluated_policy_reward = np.NINF
        all_rewards = []
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.N, self.p, self.nr_features+1))
            total_rewards = np.empty((self.N, 2))
            for k in range(self.N):
                mj_plus = (linear_policy + self.v * deltas[k])
                mj_minus = (linear_policy - self.v * deltas[k])
                total_reward_plus, v1_plus_sequence = self.perform_rollouts_v1_rff(self.H, self.env, mj_plus)
                total_reward_minus, v1_minus_sequence = self.perform_rollouts_v1_rff(self.H, self.env, mj_minus)
                total_rewards[k] = (total_reward_plus, total_reward_minus)
            sorted_ids = sort_max_index_reversed(total_rewards)

            sum_b_best_rewards = 0
            b_best_rewards = np.empty((2 * self.b))

            for i in range(self.b):
                id_x = sorted_ids[i]
                sum_b_best_rewards += (total_rewards[id_x][0] - total_rewards[id_x][1]) * deltas[id_x]
                b_best_rewards[2 * i] = total_rewards[id_x][0]
                b_best_rewards[2 * i + 1] = total_rewards[id_x][1]
            print(linear_policy)
            std_rewards = np.std(b_best_rewards)
            linear_policy = linear_policy + (self.alpha / (self.b * std_rewards)) * sum_b_best_rewards
            print(linear_policy)
            evaluated_policy_reward = evaluate_policy_v1_rff(self.H, self.env, linear_policy, self.list_of_features)
            print(evaluated_policy_reward)
            all_rewards.append(evaluated_policy_reward)
            if evaluated_policy_reward > max_evaluated_policy_reward:

                np.save(self.path + '/linear_policy.npy', linear_policy)
                np.save(self.path + '/features.npy', self.list_of_features)
                np.save(self.path + '/training_rewards.npy', all_rewards)
                np.save(self.path + '/hyper_params.npy', self.hyperparams)
                max_evaluated_policy_reward = evaluated_policy_reward
            best_total_reward = max(total_rewards[0])




    def ars_v2(self):
        # M = np.zeros(shape=(self.p, self.n))
        all_x_rewards = []
        M = np.zeros(shape=(self.p, self.n))
        max_evaluated_policy_reward = np.NINF
        all_rewards = []
        for epoch in range(self.epochs):
            deltas = np.random.standard_normal(size=(self.N, self.p, self.n))
            total_rewards = np.zeros((self.N, 2))
            sigma_rooted = self.compute_sigma_rooted()
            for k in range(self.N):
                mj_plus = (M + self.v * deltas[k])
                mj_minus = (M - self.v * deltas[k])

                total_reward_plus, v1_plus_sequence = self.perform_rollouts_v2(self.H, self.env, mj_plus, self.mean_of_encountered_states_pre_update, sigma_rooted)
                total_reward_minus, v1_minus_sequence = self.perform_rollouts_v2(self.H, self.env, mj_minus, self.mean_of_encountered_states_pre_update, sigma_rooted)

                total_rewards[k] = (total_reward_plus, total_reward_minus)

            sorted_ids = sort_max_index_reversed(total_rewards)

            sum_b_best_rewards = 0
            b_best_rewards = np.empty((2 * self.b))

            for i in range(self.b):
                id_x = sorted_ids[i]
                sum_b_best_rewards += (total_rewards[id_x][0] - total_rewards[id_x][1]) * deltas[id_x]
                b_best_rewards[2 * i] = total_rewards[id_x][0]
                b_best_rewards[2 * i + 1] = total_rewards[id_x][1]

            # print(M)
            std_rewards = np.std(b_best_rewards)
            print('std', std_rewards)
            if std_rewards == 0.0:
                print('now')
                std_rewards = 0.0001
            print(M)
            print((self.alpha / (self.b * std_rewards)))
            M = M + (self.alpha / (self.b * std_rewards)) * sum_b_best_rewards
            # print(M)
            # M = M + self.alpha * sum_b_best_rewards
            print(M)
            self.sigma_diag_pre_update = self.sigma_diag_post_update
            self.mean_of_encountered_states_pre_update = self.mean_of_encountered_states_post_update
            evaluated_policy_reward = evaluate_policy_v2(self.env, M, self.compute_sigma_rooted(), self.mean_of_encountered_states_post_update, self.H)
            all_rewards.append(evaluated_policy_reward)
            if evaluated_policy_reward > max_evaluated_policy_reward:
                np.save(self.path + '/M.npy', M)
                np.save(self.path + '/sigma_rooted.npy', self.compute_sigma_rooted())
                np.save(self.path + '/mean.npy', self.mean_of_encountered_states_post_update)
                np.save(self.path + '/training_rewards.npy', all_rewards)
                np.save(self.path + '/hyper_params.npy', self.hyperparams)

                max_evaluated_policy_reward = evaluated_policy_reward
            # best_total_reward = max(total_rewards[0])
            if epoch % self.eval_step == 0 and epoch != 0:
                print(epoch)
                all_x_rewards.append(evaluate_policy_v2(self.env, M, self.compute_sigma_rooted(), self.mean_of_encountered_states_post_update, 1000000))
                print(all_x_rewards)
                print('----')
            print(evaluated_policy_reward)
            print('----------')
        return all_x_rewards


    def perform_rollouts_v2(self, H, env, M, mean, sigma_rooted):
        total_reward = 0
        action_reward_sequence = []
        state = env.reset()
        self.update_mean_std(state)
        for i in range(H):
            action = np.matmul(np.matmul(M, sigma_rooted), (state-mean))
            next_state, reward, done, _ = env.step(action)
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

    def perform_rollouts_v1_rff(self, H, env, weights):
        total_reward = 0
        action_reward_sequence = []

        state = env.reset()
        for i in range(H):
            state = self.transform_state(state)
            action = np.matmul(weights, state)[0]
            next_state, reward, done, _ = env.step(action)

            action_reward_sequence.append([action, reward])
            # print(reward)
            state = next_state
            total_reward += reward
            if done:
                break
        return total_reward, action_reward_sequence

    def transform_state(self, state):
        transformed_state = np.zeros(len(self.list_of_features)+1)
        for i in range(len(self.list_of_features)):
            transformed_state[i] = np.cos(np.matmul(state, self.list_of_features[i]))
        transformed_state[len(self.list_of_features)] = 0.5
        return transformed_state[np.newaxis].T

    def init_random_fourier_functions(self, features_nr):
        for i in range(features_nr):
            self.list_of_features.append(np.random.randint(features_nr+1, size=self.n))


def perform_rollouts_v1(H, env, linear_policy):
    total_reward = 0
    action_reward_sequence = []
    state = env.reset()
    for i in range(H):
        action = np.matmul(linear_policy, state[np.newaxis].T)[0]
        next_state, reward, done, _ = env.step(action)
        action_reward_sequence.append([action, reward])
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward, action_reward_sequence


def sort_max_index_reversed(arr):
    n = len(arr)
    tmp = np.empty(n)
    for i in range(n):
        tmp[i] = max(arr[i])
    return np.argsort(tmp)[::-1]


def evaluate_policy_v2(env, M, sigma_rooted, mean, H, render=False):
    total_reward = 0
    action_reward_sequence = []
    state = env.reset()
    for i in range(H):
        action = np.matmul(np.matmul(M, sigma_rooted), (state - mean))
        next_state, reward, done, _ = env.step(action)
        action_reward_sequence.append([action, reward])
        state = next_state
        total_reward += reward
        if render:
            env.render()
        if done:
            break
    return total_reward

def evaluate_policy_v1_rff(H, env, weights, features, render=False):
    total_reward = 0
    action_reward_sequence = []

    state = env.reset()
    for i in range(H):
        state = transform_state(state, features)
        action = np.matmul(weights, state)[0]
        next_state, reward, done, _ = env.step(action)

        action_reward_sequence.append([action, reward])
        # print(reward)
        state = next_state
        total_reward += reward
        if render:
            env.render()
        if done:
            break
    return total_reward

def evaluate_policy_v1(H, env, weights, render=False):
    total_reward = 0
    state = env.reset()
    for i in range(H):
        action = np.matmul(weights, state[np.newaxis].T)[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if render:
            env.render()
        if done:
            break
    return total_reward


def transform_state(state, features):
        transformed_state = np.zeros(len(features)+1)
        for i in range(len(features)):
            transformed_state[i] = np.cos(np.matmul(state, features[i]))
        transformed_state[len(features)] = 0.5
        return transformed_state[np.newaxis].T

def load_policy_v1_rff(path):
    linear_policy = np.load(path+'/linear_policy.npy')
    features = np.load(path+'/features.npy')

    return linear_policy, features

def load_policy_v1(path):
    linear_policy = np.load(path+'/M.npy')
    return linear_policy


def load_policy_v2(path):
    M = np.load(path+'/M.npy')
    sigma_rooted = np.load(path+'/sigma_rooted.npy')
    mean = np.load(path + '/mean.npy')
    return M, sigma_rooted, mean

# def save_policy_v1_rff(path, linear_policy, list_of_features, all_rewards)
#     np.save(path + '/linear_policy.npy', linear_policy)
#     np.save(path + '/features.npy', list_of_features)
#     np.save(path + '/training_rewards.npy', all_rewards)
#     # np.save(path + '/hyper_params.npy', hyperparams)
#     # max_evaluated_policy_reward = evaluated_policy_reward