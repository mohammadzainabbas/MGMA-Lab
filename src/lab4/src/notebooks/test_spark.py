from torch.distributions import MultivariateNormal,Categorical
from torch.optim import Adam
import pandas as pd

import jax
import jax.numpy as jnp
import jumanji
from jumanji.wrappers import AutoResetWrapper

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

### use to flatten to observation 
def flatten(obs):
  p=[]  # obs = timestep.observation 
  p = np.append(obs.ems.x1,obs.ems.x2)
  p = np.append(p,obs.ems.y1)
  p = np.append(p,obs.ems.y2)
  p = np.append(p,obs.ems.z1)
  p = np.append(p,obs.ems.z2)
  p = np.append(p,obs.ems_mask.flatten())
  p = np.append(p,obs.items.x_len)
  p = np.append(p,obs.items.y_len)
  p = np.append(p,obs.items.z_len)
  p = np.append(p,obs.items_mask.flatten())
  p = np.append(p,obs.items_placed.flatten())
  return p 
class FeedForwardNN(nn.Module):
  def __init__(self):
    super(FeedForwardNN, self).__init__()
  def __init__(self, in_dim, out_dim):
    super(FeedForwardNN, self).__init__()
    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, out_dim)
  def forward(self, obs):
    if isinstance(obs, np.ndarray): obs = torch.tensor(obs, dtype=torch.float)  
    activation1 = F.relu(self.layer1(obs))
    output = F.relu(self.layer2(activation1))
    return output
class PPO:
  def __init__(self,env):
    self._init_hyperparameters()
    self.env = env
    self.obs_dim = 380
    self.act_dim = 800
    self.actor = FeedForwardNN(self.obs_dim,self.act_dim)
    self.critic = FeedForwardNN(self.obs_dim,1)
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
    self.cov_mat = torch.diag(self.cov_var)
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

  def _init_hyperparameters(self):
    self.timesteps_per_batch = 2  #4800            # timesteps per batch
    self.gamma = 0.95
    self.n_updates_per_iteration = 5
    self.clip = 0.2 # As recommended by the paper
    self.lr = 0.005

  def compute_rtgs(self, batch_rews):
    batch_rtgs = []
    for ep_rews in reversed(batch_rews):
      discounted_reward = 0 # The discounted reward so far
      for rew in reversed(ep_rews):
        discounted_reward = rew + discounted_reward * self.gamma
        batch_rtgs.insert(0, discounted_reward)
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    return batch_rtgs

  def get_action(self,obs,action_tensor):
     p1 = self.actor(obs)
     indices = torch.nonzero(action_tensor)  #getting valid indicies 
     p2 = p1[indices] # getting proper valid action probablities
     p2= torch.reshape(p2,(-1,))  # reshapping it to make it 1D
     probs = Categorical( logits = p2) #logits un-normalized probablities 
     a=probs.sample() #actions
     action_id = indices[a.item()].item()  # valid action value 
     return action_id, a, probs.log_prob(a)
  def rollout(self):
    batch_obs, batch_acts, batch_log_probs, batch_rews, batch_rtgs, batch_lens = [], [], [], [], [], []          
    step_fn = jax.jit(self.env.step)
    reset_fn = jax.jit(self.env.reset)
    t = 0 
    while t < self.timesteps_per_batch:
      # Rewards this episode
      ep_rews = []
      key = jax.random.PRNGKey(0)
      state, timestep = reset_fn(key)
      ep_t = 0
      rew = 0.0
      while rew == 0.0:
        # Increment timesteps ran this batch so far
        t += 1
        # Collect observation
        obs = flatten(timestep.observation)
        batch_obs.append(obs)
        num_ems, num_items = self.env.action_spec().num_values
        action_mask = timestep.observation.action_mask.flatten()
        action_tensor = torch.tensor(np.array(action_mask),dtype=torch.float)
        ems_item_id, action_,log_prob  = self.get_action(obs,action_tensor)
        ems_id, item_id = jnp.divmod(ems_item_id, num_items)
        # Wrap the action as a jax array of shape (2,)
        action = jnp.array([ems_id, item_id])
        state,timestep = step_fn(state, action)
        rew = np.array(timestep.reward.flatten())[0]
        # Collect reward, action, and log prob
        ep_rews.append(rew)
        #print("reward ", rew)
        batch_acts.append(action_)
        
        #print("action_  ",action_)
        batch_log_probs.append(log_prob)
        #print("log_probs ", log_prob)
        ep_t += 1
      # Collect episodic length and rewards
      batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
      batch_rews.append(ep_rews) 
      # Reshape data as tensors in the shape specified before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    batch_rtgs = self.compute_rtgs(batch_rews)
    return batch_obs, batch_acts,batch_log_probs, batch_rtgs, batch_lens,rew

  def learn(self, total_timesteps):
    t_so_far = 0 # Timesteps simulated so far
    episode_reward = []
    while t_so_far < total_timesteps:              # ALG STEP 2
      batch_obs, batch_acts,batch_log_probs, batch_rtgs, batch_lens,rew = self.rollout()
      episode_reward.append(rew)
      # Calculate how many timesteps we collected this batch   
      t_so_far += np.sum(batch_lens)
      V, _ = self.evaluate(batch_obs, batch_acts)
      # Calculate advantage
      A_k = batch_rtgs - V.detach()
      # Normalize advantages
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
      for i in range(self.n_updates_per_iteration):
        # Calculate V_phi and pi_theta(a_t | s_t)    
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
        # Calculate ratios
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean()
        # Calculate gradients and perform backward propagation for actor 
        # network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        critic_loss = nn.MSELoss()(V, batch_rtgs)
        # Calculate gradients and perform backward propagation for critic network    
        self.critic_optim.zero_grad()    
        critic_loss.backward()    
        self.critic_optim.step()
    return episode_reward
    
  def evaluate(self, batch_obs,batch_acts):
    # Query critic network for a value V for each obs in batch_obs.
    V = self.critic(batch_obs).squeeze()
    # Calculate the log probabilities of batch actions using most recent actor network.
    # This segment of code is similar to that in get_action()
    mean = self.actor(batch_obs)
    dist = Categorical(mean)
    log_probs = dist.log_prob(batch_acts)
    # Return predicted values V and log probs log_probs
    return V, log_probs
 
  