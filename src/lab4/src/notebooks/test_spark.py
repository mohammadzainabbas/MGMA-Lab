import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

import jax.numpy as np
from jax import grad, jit, vmap
from jax.ops import index_add, index_update

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

        return critic_fn, {'layer1': np.random.randn(self.obs_dim, 64),
