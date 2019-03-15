import matplotlib.pyplot as plt
import numpy as np

def plt_expected_cum_reward(path, expected_reward, eval_step):
    """
    path to store evaluation plots

    :param expected_reward:
    :return:
    """

    fig, ax = plt.subplots()
    ax.plot(expected_reward, color='red')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Expected Cumulative Reward')
    ax.grid(alpha=0.5, linestyle='-')


    fig.savefig(path+'/reward.png')
    plt.close('all')