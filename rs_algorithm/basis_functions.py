import numpy as np

class PolynomialBasis():

    def __init__(self, dim_state, degree):
        """
        Creates polynomial feature object

        :param dim_state: state dimension
        :param num_actions: number of action
        :param degree: degree of polynomial
        """
        self.num_states = dim_state
        self.num_features = (dim_state * degree + 1)
        self.degree = degree

    def evaluate(self, state):
        """

        :param state:
        :return:
        """
        feature = np.zeros(shape=(self.num_features, 1))
        pos = 0
        feature[pos] = 1
        pos += 1
        for s in state:
            for d in (1, self.degree):
                feature[pos] = np.power(s, d)
                pos += 1
        return feature


    def evaluate_(self, state, action_index):
        """
        This function computes the feature matrix phi

        :param state: state = (x, cos_theta, sin_theta, x_dot, phi_dot)
        :param action_index: index of action from discretized actions
        :return: feature vector phi
        """
        offset = action_index * (self.num_states * self.degree + 1)
        k = self.num_features
        feature = np.zeros(shape=(k, 1))

        feature[offset] = 1
        offset += 1
        for s in state:
            for d in range(1, self.degree + 1):
                feature[offset] = np.power(s, d)
                offset += 1

        return feature

class FourierBasis():

    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.num_features = np.power(state_dim, state_dim) + 1

    def evaluate(self, state):
        """

        :param state:
        :return:
        """
        feature = np.zeros(shape=(self.num_features, 1))
        pos = 0
        feature[pos] = 1
        pos += 1
        for i in range(len(state)):
            for j in range(len(state)):
                feature[pos] = (np.cos(2 * i * np.pi) / len(state)) * state[j]
                feature[pos+1] = (np.sin(2 * i * np.pi) / len(state)) * state[j]
                pos += 2

        return feature

