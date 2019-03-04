import gym
from quanser_robots import GentlyTerminating
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from ppo_old.models.actorcritic import PolicyNetwork
from ppo_old.models.actorcritic import ValueNetwork
from ppo_old.models.actorcritic import compute_general_advantage_estimate
'''
def create_ppo_actor_batches(mini_batch_size, ):
    """ This method creates batches of mini batches for ppo_old
    :param mini_batch_size: size of mini batches
    :param """
    batch_size =
    for _ in range(batch_size // mini_batch_size):
'''


def ppo_update(epochs, states, actions, old_log_probs, advantage_estimates, adv, optimizer_actor, optimizer_critic, policy_net, value_net, eps, c1=0.001):
    """ This method performs the proximal policy optimization algorithm.
    :param epochs: number of ppo_old iterations
    :param states: list of states. length equals trajectory length
    :param actions: list of actions.
    :param advantage_estimates: advantage estimates computed by gae
    :param policy_net: forward propagation of policy network
    :param value_net: forward propagation of value network
    :param eps: clipping parameter for clipped surrogate objective
    :param c1: coefficient for squared-error loss
    :param c2: coefficient for entropy bonus (entropy bonus ensures sufficient exploration)
    """
    for _ in range(epochs):
        for i in range(len(states)):
            # vector of vectors of states
            dist = policy_net(states[i])
            value = value_net(states[i])
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(actions[i]) # take log probability because of faster computation

            policy_ratio = (log_probs - old_log_probs[i]).exp()
            cl1 = policy_ratio * adv[i]
            cl2 = torch.clamp(policy_ratio, 1.0 - eps, 1.0 + eps) * adv[i]
            clip = torch.min(cl1, cl2)

            actor_loss = clip.mean() # or value network loss
            critic_loss = (advantage_estimates[i] - value).pow(2).mean()
            #print("Critic Loss:", critic_loss)

            #loss = 0.5 * critic_loss - actor_loss + c1 * entropy
            loss = critic_loss - actor_loss + c1 * entropy
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_actor.step()
            optimizer_critic.step()

def run_ppo(epochs, env_platform, trajectory_size, vis=False, plot_reward=False):
    """ This method computes ppo_old on
    :param epochs: number of epochs to run ppo_old
    :param env_platform: name of gym environment
    :param vis: rendering simulation if true
    :param plot_reward: plots reward if true
    """
    env = GentlyTerminating(env_platform)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    actor = PolicyNetwork(num_inputs, num_outputs, 64)
    critic = ValueNetwork(num_inputs, 64)

    state = env.reset()

    # initialize optimizers for each network
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)
    done = False
    plot_rewards = []
    action_list = []
    iteration = 0
    action_list = []

    for e in range(epochs):
        if done:
            done = False
            state = env.reset()

        while not done:

            ppo_epochs = 5
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            entropy = 0
            total_reward = 0
            mean_total_reward = 0
            penalize = done * 2 # penalize reward if physics are violated

            for i in range(trajectory_size):
                # state = torch.FloatTensor(state).reshape(2)

                state = torch.FloatTensor(state)
                dist = actor.forward(state)
                value = critic.forward(state)

                action = dist.sample()[0]

                action_list.append(action)
                next_state, reward, done, info = env.step(action.cpu().numpy())
                if reward > 1.9:
                    reward *= 2.

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)

                rewards.append(torch.FloatTensor([reward - penalize]))

                states.append(state)

                actions.append(action)

                state = next_state
                total_reward += reward

                if vis: env.render()
                #env.reset()
            mean_total_reward += total_reward
            if iteration % 10 == 0:
                plot_rewards.append(mean_total_reward/10)

            print("Reward: {} Iteration: {} Epoch: {}".format(total_reward, iteration, e))
            print("----------------------")
            iteration += 1


            # next_state = torch.FloatTensor(next_state).reshape(2)
            next_state = torch.FloatTensor(next_state)
            next_value = critic.forward(next_state)


            advantages_estimates = compute_general_advantage_estimate(rewards, values, next_value, 0.99, 0.95)

            # detach from current graph

            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states).split(num_inputs)
            actions = torch.cat(actions)
            advantage = advantages_estimates - values

            ppo_update(ppo_epochs, states, actions, log_probs, advantages_estimates, advantage, optimizer_actor,
                       optimizer_critic, actor.forward, critic.forward, 0.2)

    if plot_reward:
        plt.plot(plot_rewards)
        plt.show()
