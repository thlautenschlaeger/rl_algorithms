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

    rew = np.empty(300)
    for i in range(10):
        file = np.load('/Users/thomas/Seafile/PersonalCloud/informatik'
                       '/master/semester_2/reinforcement_learning/data/3rr/tr/'
                       'transition_rewards' + str(i) + '.npy')
        rew = np.vstack((rew, file))

    rewards = rewards[1:10]
    rew = rew[0:11]

    std = np.std(rewards, axis=0)
    stds = np.std(rew, axis=0)
    rewards = np.mean(rewards, axis=0)
    rewardss = np.mean(rew, axis=0)


    choose = np.arange(0, len(rewards) + 1, step=15)
    choose[-1] = choose[-1] -1
    # x = np.arange(300)[::steps]
    # rewards = rewards[::steps]
    # std = std[::steps]
    x = np.arange(300)[choose]
    rewards = rewards[choose]
    rewardss = rewardss[choose]
    std = std[choose]
    stds = stds[choose]

    fig, ax = plt.subplots()

    # ax.fill_between(x=x, y1=rewards-std, y2=rewards+std)

    rr = ax.errorbar(x=x, y=rewards, yerr=std, linestyle=None, fmt='-o', markersize='3.5', solid_capstyle='projecting', capsize=3)
    ss = ax.errorbar(x=x, y=rewardss, yerr=stds, linestyle=None, fmt='-o', markersize='3.5', solid_capstyle='projecting',
                     capsize=3)

    ax.set_xlabel('Transition')
    ax.set_ylabel('Accumulated Reward')
    ax.set_title('PPO Qube')
    ax.grid(alpha=0.5, linestyle='-')
    plt.legend([rr, ss], ["real robot", "simulation"])

    path = '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures'
    plt.savefig(path+'/transition.pdf',format='pdf')
    plt.show()

def plot_cart_rrr():
    check = torch.load('/Users/thomas/Desktop/noob/cart6_save/checkpoint/save_file.pt')
    eval_rewards = check['eval_rewards']
    eval_std = check['eval_rewards_std']

    x = np.arange(0, 50, 50/ len(eval_std))
    # x = range(len(eval_rewards))
    rr = plt.errorbar(x=x, y=eval_rewards, yerr=eval_std, linestyle=None, fmt='-o', markersize='3.5',
                     solid_capstyle='projecting',
                     capsize=3)

    plt.grid(alpha=0.5, linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Expected Reward')
    plt.title('PPO CartpoleRR')
    plt.savefig('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures/training_rr_cartpole.pdf', format='pdf')
    plt.show()


def plot_cart_rr():
    def zero_to_nan(values):
        """Replace every 0 with 'nan' and return a copy."""
        return [float('nan') if x == 0 else x for x in values]

    rewards = np.zeros(10000)
    for i in range(3,10):

        file = np.load('/Users/thomas/Desktop/noob/cart55/tr/transition_rewards'+str(i)+'.npy')
        lel = 10000 - len(file)
        lel = np.zeros(lel)
        file = np.concatenate((file, lel))
        # file = zero_to_nan(file)
        rewards = rewards + file

    rrewards = rewards / 5

    rewardsss = np.empty(10000)
    for i in range(0, 10):
        file = np.load('/Users/thomas/Desktop/noob/cart5/tr/transition_rewards' + str(i) + '.npy')
        rewardsss = np.vstack((rewardsss, file))

    srewards = rewardsss[1:11]

    sstd = np.std(srewards, axis=0)
    rstd = np.ones(len(rrewards))
    # rewards = np.mean(rewards, axis=0)[0:2000]
    srewards = np.mean(srewards, axis=0)
    choose = np.arange(0, len(srewards) + 1, step=500)
    choose[-1] = choose[-1] - 1
    x = np.arange(10000)[choose]

    rrewards = rrewards[choose]

    srewards = srewards[choose]
    sstd = sstd[choose]
    rstd = rstd[choose]

    fig, ax = plt.subplots()

    rr = ax.errorbar(x=x, y=rrewards, yerr=rstd, linestyle=None, fmt='-o', markersize='3.5', solid_capstyle='projecting',
                     capsize=3)
    ss = ax.errorbar(x=x, y=srewards, yerr=sstd, linestyle=None, fmt='-o', markersize='3.5', solid_capstyle='projecting',
                     capsize=3)

    ax.grid(alpha=0.5, linestyle='-')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Accumulated Reward')
    ax.set_title('PPO Cartpole')

    plt.legend([ss, rr], ["simulation", "real robot"])
    plt.savefig('/Users/thomas/Seafile/PersonalCloud/informatik/master/'
                'semester_2/reinforcement_learning/figures/compare_rr_sim.pdf', format='pdf')

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

def plot_rs_expected_rewards():
    cart_v0_0 = np.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/rs_data/v0-cartpole1/eval_rewards.npy')
    cart_v0_1 = np.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/rs_data/v0-cartpole2/eval_rewards.npy')

    rewards1 = np.vstack((cart_v0_0, cart_v0_1))

    std = np.std(rewards1, axis=0)
    rewards1 = np.mean(rewards1, axis=0)

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5, linestyle='-')

    choose = np.arange(len(rewards1) + 1, step=25)
    choose[-1] = choose[-1] - 1
    x = np.arange(len(rewards1))[choose]
    plot_rewards = rewards1[choose]
    std = std[choose]

    ax.errorbar(x, y=plot_rewards, yerr=std, linestyle=None, fmt='o', markersize='5.5',
        solid_capstyle='projecting', capsize=5)


    plt.show()

def plot_reg():
    reg1 = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/data/layer_normed2/checkpoint/save_file.pt')
    reg2 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/data/layer_normed2/checkpoint/save_file.pt')

    noreg1 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/data/qube2/checkpoint/save_file.pt')
    noreg2 = torch.load(
        '/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/data/qube1/checkpoint/save_file.pt')


    reg1 = reg1['eval_rewards']
    reg2 = reg2['eval_rewards']
    no1  = noreg1['eval_rewards']
    no2 = noreg2['eval_rewards']

    rewards = np.vstack((reg1, reg2))
    rewardss = np.vstack((no1, no2))

    std1 = np.std(rewards, axis=0)
    rewards1 = np.mean(rewards, axis=0)

    std2 = np.std(rewardss, axis=0)[0:1000]
    rewards2 = np.mean(rewardss, axis=0)[0:1000]

    fig, ax = plt.subplots()
    ax.grid(alpha=0.5, linestyle='-')

    choose = np.arange(len(rewards1) + 1, step=10)
    choose[-1] = choose[-1] - 1
    x = np.arange(len(rewards1))[choose]
    plot_rewards = rewards1[choose]
    std1 = std1[choose]

    rewardss = rewards2[choose]
    std2 = std2[choose]


    reg = ax.errorbar(x=x, y=plot_rewards, yerr=std1, linestyle=None, fmt='-o', markersize='5.5',
                      solid_capstyle='projecting', capsize=5)
    noreg = ax.errorbar(x=x, y=rewardss, yerr=std2, linestyle=None, fmt='-o', markersize='5.5',
                      solid_capstyle='projecting', capsize=5)

    ax.set_xlabel('Episode')
    ax.set_xlabel('Expected Reward')
    ax.set_title('PPO Qube')

    plt.legend([reg, noreg], ["layer norm", "no layer norm"])

    plt.savefig('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures/regualarized.pdf', format='pdf')


    plt.show()

def expect_levitation():
    check = torch.load('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/project/rl_algorithms/ppo_algorithm/data/Levitation-v1_2019-03-13_19-39-37/checkpoint/save_file.pt')
    expected_rewards = check['eval_rewards']
    std = check['eval_rewards_std']

    x = np.arange(len(expected_rewards))
    choose = np.arange(len(expected_rewards) + 1, step=10)
    choose[-1] = choose[-1] - 1

    expected_rewards = expected_rewards[choose]
    std = std[choose]

    x = x[choose]
    plt.errorbar(x=x, y=expected_rewards, yerr=std, linestyle=None, fmt='-o', markersize='5.5',
                      solid_capstyle='projecting', capsize=5)

    plt.xlabel('Episode')
    plt.ylabel('Expected Reward')
    plt.title('PPO Levitation')
    plt.grid(alpha=0.5, linestyle='-')
    plt.savefig('/Users/thomas/Seafile/PersonalCloud/informatik/master/semester_2/reinforcement_learning/figures/levitation.pdf', format='pdf')
    plt.show()


expect_levitation()
# plot_reg()
# plot_cart_rrr()
# plot_rs_expected_rewards()


# plot_expected_cumulative_rewards_()
# plot_expected_entropy()


# plot_rr()
