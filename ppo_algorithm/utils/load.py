import torch
import matplotlib.pyplot as plt
import numpy as np

# lel = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/Qube-v0_2019-03-12_23-33-17/checkpoint/save_file.pt', map_location='cpu')

# rewards = lel['eval_rewards']
# std = lel['eval_rewards_std']
# entropy = lel['entropy']



# plt.plot(entropy)
# plt.errorbar(np.arange(len(rewards)), rewards, yerr=std)
# plt.show()

def plot_rr():
    """
    Plots transition rewards from RR Cube
    :return:
    """
    rewards = np.empty(300)
    for i in range(10):
        if i !=7:
            file = np.load('/Users/thomas/Seafile/PersonalCloud/informatik/master'
                               '/semester_2/reinforcement_learning/'
                               'project/rl_algorithms/data/3copy/transition_rewards'+str(i)+'.npy')
            rewards = np.vstack((rewards, file))
        # print('Number: {} length {}'.format(i, len(file)))

    rewards = rewards[1:10]

    std = np.std(rewards, axis=0)
    rewards = np.mean(rewards, axis=0)


    choose = np.arange(0, len(rewards) + 1, step=15)
    choose[-1] = choose[-1] -1
    # x = np.arange(300)[::steps]
    # rewards = rewards[::steps]
    # std = std[::steps]
    x = np.arange(300)[choose]
    rewards = rewards[choose]
    std = std[choose]

    fig, ax = plt.subplots()

    # ax.fill_between(x=x, y1=rewards-std, y2=rewards+std)

    ax.errorbar(x=x, y=rewards, yerr=std, linestyle=None, fmt='-o', markersize='3.5', solid_capstyle='projecting', capsize=3)
    ax.set_xlabel('Transitions')
    ax.set_ylabel('Accumulated reward')
    ax.set_title('Average transition rewards on QubeRR-v0')
    ax.grid(alpha=0.5, linestyle='-')

    path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures'
    plt.savefig(path+'/transition.pdf',format='pdf')
    plt.show()

def plot_expected_cumulative_rewards():
    file = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube3/checkpoint/save_file.pt')

    rewards = file['eval_rewards']
    std = file['eval_rewards_std']
    choose = np.arange(len(rewards) + 1, step=15)
    choose[-1] = choose[-1] -1
    x = np.arange(len(rewards))[choose]
    plot_rewards = rewards[choose]
    std = std[choose]

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5, linestyle='-')

    ax.errorbar(x=x, y=plot_rewards, yerr=std, linestyle=None, fmt='-o', markersize='3.5',
                solid_capstyle='projecting', capsize=3)

    plt.show()

def plot_expected_cumulative_rewards_():
    file1 = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube1/checkpoint/save_file.pt')
    file2 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube2/checkpoint/save_file.pt')
    file3 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube3/checkpoint/save_file.pt')
    file4 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube4/checkpoint/save_file.pt')

    # rewards = file['eval_rewards']
    rewards = file1['eval_rewards']
    rewards = np.vstack((rewards,file2['eval_rewards']))
    rewards = np.vstack((rewards,file3['eval_rewards']))
    rewards = np.vstack((rewards,file4['eval_rewards']))

    std = np.std(rewards, axis=0)
    rewards = np.mean(rewards, axis=0)



    # std = file['eval_rewards_std']
    choose = np.arange(len(rewards) + 1, step=25)
    choose[-1] = choose[-1] -1
    x = np.arange(len(rewards))[choose]
    plot_rewards = rewards[choose]
    std = std[choose]

    fig, ax = plt.subplots()
    # ax.set_xlim(0, 5000);
    ax.grid(alpha=0.5, linestyle='-')
    # kek = np.arange(0, 5000, 1000)
    ax.set_xticklabels(np.arange(-1000, 7000, 1000))
    ax.errorbar(x=x, y=plot_rewards, yerr=std, linestyle=None, fmt='-o', markersize='5.5',
                solid_capstyle='projecting', capsize=5)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Expected Return')

    ax.set_title('Expected return on Qube-v0')
    path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures'
    plt.savefig(path + '/qube_expected_return.pdf', format='pdf')





    plt.show()

def plot_expected_entropy():
    file1 = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube1/checkpoint/save_file.pt')
    file2 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube2/checkpoint/save_file.pt')
    file3 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube3/checkpoint/save_file.pt')
    file4 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/data/qube4/checkpoint/save_file.pt')

    # rewards = file['eval_rewards']
    rewards = file1['entropy']
    rewards = np.vstack((rewards,file2['entropy']))
    rewards = np.vstack((rewards,file3['entropy']))
    rewards = np.vstack((rewards,file4['entropy']))

    std = np.std(rewards, axis=0)
    rewards = np.mean(rewards, axis=0)


    choose = np.arange(len(rewards) + 1, step=250)
    choose[-1] = choose[-1] -1
    x = np.arange(len(rewards))[choose]
    plot_rewards = rewards[choose]
    std = std[choose]

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5, linestyle='-')

    ax.errorbar(x=x, y=plot_rewards, yerr=std, linestyle=None, fmt='-o', markersize='5.5',
                solid_capstyle='projecting', capsize=5)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Entropy')

    ax.set_title('Average entropy on Qube-v0')
    path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures'
    plt.savefig(path+'/qube_entropy.pdf', format='pdf')
    plt.show()


plot_expected_cumulative_rewards_()
# plot_expected_entropy()

# plot_rr()
