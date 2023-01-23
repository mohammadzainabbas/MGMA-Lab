import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import jax.numpy as np
from jax import grad, jit, vmap
from jax.ops import index_add, index_update

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
