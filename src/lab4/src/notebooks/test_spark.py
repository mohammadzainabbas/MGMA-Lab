import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import jax.numpy as np
from jax import grad, jit, vmap
from jax.ops import index_add, index_update
import jumanji

def flatten(obs):
    """
    This function takes in the observation object and concatenates the relevant fields of the object into a single 1D array, which is returned.
    """
    return np.concatenate((obs.ems.x1, obs.ems.x2, obs.ems.y1, obs.ems.y2, obs.ems.z1, obs.ems.z2, obs.ems_mask.flatten(), obs.items.x_len, obs.items.y_len, obs.items.z_len, obs.items_mask.flatten(), obs.items_placed.flatten()))

"""
In the FeedForwardNN class, use the torch.no_grad() context when converting the numpy array to a PyTorch tensor to prevent the creation of unnecessary gradients, which can save memory and computation time.

In the PPO class, the compute_rtgs method can be optimized by using the torch.cumsum function to compute the cumulative sum of the rewards in reverse order. This will reduce the number of iterations required to compute the rewards.

In the get_action method, use the torch.gather function instead of the torch.nonzero function to retrieve the valid indices. This will be faster as it avoids creating a new tensor.

Instead of using a while loop in the rollout method, use a for loop with a range of self.timesteps_per_batch to iterate over the number of required timesteps.

In the rollout method, consider using JIT compilation (e.g. jax.jit) on the step_fn and reset_fn function calls to speed up the execution of the environment's step and reset methods.

Lastly, you can consider using the torch multiprocessing library to parallelize the environment steps and the network computations.

Another way to optimize the code is to use the torch.optim library's built-in optimization algorithms instead of manually updating the model's parameters. This can reduce the amount of code and make the training process more efficient.

Also, you can use the torch.nn.DataParallel wrapper around the actor and critic network models, this will allow you to train on multiple GPUs.

One more thing, you can use the torch.autograd.profiler to profile your code and find the bottlenecks in the code, and then optimize the specific parts of the code that take the most time.

Another way you can optimize is by using the torch.cuda library to perform computations on the GPU. This will increase the speed of the computations as the GPU is designed for parallel processing.

"""


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, out_dim)

    def forward(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float)  
        activation1 = F.relu(self.layer1(obs))
        output = F.relu(self.layer2(activation1))
        return output

class PPO:
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env
        self.obs_dim = 380
        self.act_dim = 800
        self.actor = self._init_actor()
        self.critic = self._init_critic()
        self.cov_var = np.full(self.act_dim, fill_value=0.5)
        self.cov_mat = np.diag(self.cov_var)
        self.actor_optim = self._init_optimizer()
        self.critic_optim = self._init_optimizer()

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2  #4800            # timesteps per batch
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.005

    def _init_actor(self):
        def actor_fn(params, obs):
            activation1 = np.maximum(np.dot(obs, params['layer1']), 0)
            return np.dot(activation1, params['layer2'])

        return actor_fn, {'layer1': np.random.randn(self.obs_dim, 64),
                          'layer2': np.random.randn(64, self.act_dim)}

    def _init_critic(self):
        def critic_fn(params, obs):
            activation1 = np.maximum(np.dot(obs, params['layer1']), 0)
            return np.dot(activation1, params['layer2'])

        return critic_fn, {'layer1': np.random.randn(self.obs_dim, 64), 'layer2': np.random.randn(64, 1)}
        
    def _init_optimizer(self):
        def optimizer_fn(params, grads, lr):
            for p, g in zip(params.values(), grads.values()):
                p -= lr * g
            return params

        return optimizer_fn

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return np.array(batch_rtgs)

    def get_action(self, obs, action_tensor):
        p1 = self.actor[0](self.actor[1], obs)
        indices = np.nonzero(action_tensor)  # getting valid indices
        p2 = p1[indices] # getting proper valid action probabilities
        p2 = p2.flatten() # reshaping it to make it 1D
        probs = Categorical(logits=p2) # logits un-normalized probabilities
        a = probs.sample() # actions
        action_id = indices[a.item()].item()  # valid action value
        return action_id, a, probs.log_prob(a)

    def rollout(self):
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = [], [], [], [], [], []
        for t in range(self.timesteps_per_batch):
            ep_rews = []
            state, timestep = self.env.reset()
            obs = flatten(timestep.observation)
            while not timestep.last():
                action_tensor = np.ones(self.act_dim)
                action_id, a, log_prob = self.get_action(obs, action_tensor)
                next_timestep = self.env.step(action_id)
                next_obs = flatten(next_timestep.observation)
                rew = next_timestep.reward
                ep_rews.append(rew)
                state, timestep = next_timestep
                obs = next_obs
                t += 1
                #collect data
                batch_obs.append(obs)
                batch_acts.append(action_id)
                batch_log_probs.append(log_prob)
                batch_rews.append(rew)
                batch_lens.append(t)
            batch_rtgs.append(self.compute_rtgs(ep_rews))
        #update
        self.update(batch_obs,batch_acts,batch_log_probs,batch_rtgs,batch_lens)

    def learn(self, num_iterations):
        for i in range(num_iterations):
            self.rollout()

    def evaluate(self, num_episodes):
        rewards = []
        for i in range(num_episodes):
            state, timestep = self.env.reset()
            obs = flatten(timestep.observation)
            episode_reward = 0
            while not timestep.last():
                action_tensor = np.ones(self.act_dim)
                action_id, _, _ = self.get_action(obs, action_tensor)
                next_timestep = self.env.step(action_id)
                obs = flatten(next_timestep.observation)
                episode_reward += next_timestep.reward
                state, timestep = next_timestep
            rewards.append(episode_reward)
        return rewards

    def update(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens):
        """
        In this function, we first convert the input lists into numpy arrays, then we update the actor and critic networks using the PPO algorithm.

        We use Jax's grad function to compute the gradients of the actor and critic loss functions, and jit function to speed up the computation.

        You may need to adjust the current code to make it work with Jax, as the current codebase is using torch and numpy.
        """
        obs = np.array(batch_obs)
        acts = np.array(batch_acts)
        log_probs = np.array(batch_log_probs)
        rtgs = np.array(batch_rtgs)
        lens = np.array(batch_lens)
        self.actor_optim = self._init_optimizer()
        self.critic_optim = self._init_optimizer()
        for _ in range(self.n_updates_per_iteration):
            old_log_probs = self.actor[0](self.actor[1], obs)[np.arange(len(batch_obs)), acts]
            ratios = np.exp(log_probs - old_log_probs)
            advantages = rtgs - self.critic[0](self.critic[1], obs)
            surrogate_obj = ratios * advantages
            surrogate_obj_clipped = np.minimum(ratios * advantages, np.ones_like(ratios) * self.clip)
            actor_loss = -np.mean(surrogate_obj_clipped)
            critic_loss = np.mean((rtgs - self.critic[0](self.critic[1], obs))**2)
            grads_actor = grad(actor_loss)(self.actor[1])
            grads_critic = grad(critic_loss)(self.critic[1])
            self.actor[1] = self.actor_optim(self.actor[1], grads_actor, self.lr)
            self.critic[1] = self.critic_optim(self.critic[1], grads_critic, self.lr)



# Initialize the environment
env = jumanji.make("BinPack-toy-v0")

# Initialize the PPO agent
agent = PPO(env)

# Train the agent for a specified number of iterations
num_iterations = 100
agent.learn(num_iterations)

# Evaluate the agent's performance
num_episodes = 10
rewards = agent.evaluate(num_episodes)
print(f'Average reward over {num_episodes} episodes: {np.mean(rewards)}')
