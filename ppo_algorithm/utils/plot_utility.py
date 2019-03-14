import matplotlib.pyplot as plt
import numpy as np

def plt_expected_cum_reward(mean, std, path, eval_step):
    """

    :param mean:
    :param std:
    :return:
    """

    fig, ax = plt.subplots()
    ax.plot(mean, color='red')
    n = len(mean)
    x = np.arange(n)
    ax.fill_between(x, mean-std, mean+std, facecolor='blue')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Expected Cumulative Reward')
    ax.grid(alpha=0.5, linestyle='-')


    fig.savefig(path+'/reward.png')
    plt.close('all')

def plot_transitions(mean_reward, std_reward):
    """

    :param mean_reward:
    :param std_reward:
    :return:
    """

