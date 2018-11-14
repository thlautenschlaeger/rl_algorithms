import torch
'''
def create_ppo_actor_batches(mini_batch_size, ):
	""" This method creates batches of mini batches for ppo
	:param mini_batch_size: size of mini batches
	:param """
	batch_size =
	for _ in range(batch_size // mini_batch_size):
'''


def ppo_update(epochs, states, actions, old_log_probs, advantage_estimates, adv, optimizer_actor, optimizer_critic, policy_net, value_net, eps, c1=0.001):
	""" This method performs the proximal policy optimization algorithm.
	:param epochs: number of ppo iterations
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
			log_probs = dist.log_prob(actions[i])

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