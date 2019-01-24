import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def plot_moving_avg_reward(total_rewards, trajectory_size):
    """
    This method plots the average reward as moving average

    :param total_rewards: list of total rewards
    :param trajectory_size:
    :return: plots average reward
    """
    plot_avg_reward = []
    for i in range(0, len(total_rewards), trajectory_size):
        tmp = np.mean(total_rewards[i:i + trajectory_size])
        plot_avg_reward.append(tmp)

    x = np.arange(len(total_rewards))
    xp = np.arange(start=0, stop=len(total_rewards), step=trajectory_size)
    plot_avg_reward = np.interp(x, xp, plot_avg_reward)

    plot_avg_reward = gaussian_filter1d(plot_avg_reward, sigma=2)

    plt.plot(plot_avg_reward)
    plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_running_mean(x, N):
    plt.plot(running_mean(x, N))
    plt.show()