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
    #self.layer3 = nn.Softmax()
  def forward(self, obs):
  # Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray):
      obs = torch.tensor(obs, dtype=torch.float)
      #print("inside tensor")
  
    activation1 = F.relu(self.layer1(obs))
    output = F.relu(self.layer2(activation1))
    #output = self.layer3(activation2)
    return output

class PPO:
  def __init__(self,env):
    self._init_hyperparameters()
    self.env = env
    #####################################
    self.obs_dim = 380
    self.act_dim = 800
    ######################################
    

    #initiate actor and critic
    self.actor = FeedForwardNN(self.obs_dim,self.act_dim)
    self.critic = FeedForwardNN(self.obs_dim,1)

      # Create our variable for the matrix.
    # Note that I chose 0.5 for stdev arbitrarily.
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
    
    # Create the covariance matrix
    self.cov_mat = torch.diag(self.cov_var)
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
    
    
  def _init_hyperparameters(self):
    # Default values for hyperparameters, will need to change later.
    self.timesteps_per_batch = 2  #4800            # timesteps per batch
    #self.max_timesteps_per_episode = 1600      # timesteps per episode
    self.gamma = 0.95
    self.n_updates_per_iteration = 5
    self.clip = 0.2 # As recommended by the paper
    self.lr = 0.005

  def compute_rtgs(self, batch_rews): 
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []
    #print("batch_rewards ",batch_rews)
    # Iterate through each episode backwards to maintain same order
    # in batch_rtgs
    for ep_rews in reversed(batch_rews):
      discounted_reward = 0 # The discounted reward so far
      for rew in reversed(ep_rews):
        discounted_reward = rew + discounted_reward * self.gamma
        batch_rtgs.insert(0, discounted_reward)
    # Convert the rewards-to-go into a tensor
    #print("discounted reward of batch", batch_rtgs)
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    return batch_rtgs

  def get_action(self,obs,action_tensor):
    #flatten observation  p = flatten(timestep.observation)
    # inside critic and actor by converting it to np.array(p) by actor you will get action
     p1 = self.actor(obs)
     indices = torch.nonzero(action_tensor)  #getting valid indicies 
     p2 = p1[indices] # getting proper valid action probablities
     p2= torch.reshape(p2,(-1,))  # reshapping it to make it 1D
     probs = Categorical( logits = p2) #logits un-normalized probablities 
     a=probs.sample() #actions
     #print("in get_action , shape of valid probs", p2.shape)
     # probs.log_prob(a) # softmax probablity of that action
     action_id = indices[a.item()].item()  # valid action value 
     return action_id, a, probs.log_prob(a)  

    #  ind = action_mask.index[[action_mask[0] == True]]
    #  df = pd.DataFrame(p1.detach().numpy())
    #  probs = Categorical(probs=torch.tensor(list(df.iloc[ind][0])))
    #  action = probs.sample()
    #  action_id = df.iloc[ind][0][action].index
     


   
  def rollout(self):
    # Batch data
    batch_obs = []             # batch observations
    batch_acts = []            # batch actions
    batch_log_probs = []       # log probs of each action
    batch_rews = []            # batch rewards
    batch_rtgs = []            # batch rewards-to-go
    batch_lens = []            # episodic lengths in batch
                 # for animation
    # Number of timesteps run so far this batch´
    
    step_fn = jax.jit(self.env.step)
    reset_fn = jax.jit(self.env.reset)
    t = 0 
    while t < self.timesteps_per_batch:
      # Rewards this episode
      ep_rews = []
      key = jax.random.PRNGKey(0)
      ###############################
      #jax.jit(env.reset)(key)
      state, timestep = reset_fn(key)
     
      ###############################
      ep_t = 0
      rew = 0.0
      #for ep_t in range(self.max_timesteps_per_episode):
      while rew == 0.0:
        # Increment timesteps ran this batch so far
        t += 1
        # Collect observation
        ################################################
        obs = flatten(timestep.observation)
        #obs = torch.tensor(obs, dtype=torch.float)
        batch_obs.append(obs)
        
        num_ems, num_items = self.env.action_spec().num_values
        action_mask = timestep.observation.action_mask.flatten()
        action_tensor = torch.tensor(np.array(action_mask),dtype=torch.float)
         
        
        #----------------------------------------------- get from NN
        #ems_item_id = self.get_action(obs,action_mask)
        ems_item_id, action_,log_prob  = self.get_action(obs,action_tensor)
        # -------------------------------------------------
        ems_id, item_id = jnp.divmod(ems_item_id, num_items)

        # Wrap the action as a jax array of shape (2,)
        action = jnp.array([ems_id, item_id])

        #action = torch.tensor(action, dtype=torch.float)
        #ems_item_id, action_,log_prob  = self.get_action(obs,action_mask)
        #mean = self.actor(obs)
        #dist = MultivariateNormal(mean, self.cov_mat)
        #log_prob = dist.log_prob(action)
        #batch_states.append(state)
        state,timestep = step_fn(state, action)
        rew = np.array(timestep.reward.flatten())[0]
        ##################################################
        # Collect reward, action, and log prob
        ep_rews.append(rew)
        #print("reward ", rew)
        batch_acts.append(action_)
        
        #print("action_  ",action_)
        batch_log_probs.append(log_prob)
        #print("log_probs ", log_prob)
        ep_t += 1
      # Collect episodic length and rewards
      #print("end of episode", ep_t)
      batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
      #print("episode length  ," , ep_t+1)
      batch_rews.append(ep_rews) 
      #print("episode rewards ", ep_rews)
      # Reshape data as tensors in the shape specified before returning
    #print("end of batch ")
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    #print("batch_observation ", batch_obs)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    #print("batch_action ", batch_acts)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    #print("batch log_probs ",batch_log_probs)
    # ALG STEP #4
    #print("batch_rewards ",batch_rews)
    batch_rtgs = self.compute_rtgs(batch_rews)
    #print("discounted rewards ",batch_rtgs)
    #env.render(state)
    # Return the batch data
    return batch_obs, batch_acts,batch_log_probs, batch_rtgs, batch_lens,rew

  def learn(self, total_timesteps):
    t_so_far = 0 # Timesteps simulated so far
    episode_reward = []
    #print("total timesteps ",total_timesteps, " updates per iteration ", self.n_updates_per_iteration)
    while t_so_far < total_timesteps:              # ALG STEP 2
      # Increment t_so_far somewhere below
      # ALG STEP 3
      batch_obs, batch_acts,batch_log_probs, batch_rtgs, batch_lens,rew = self.rollout()
      episode_reward.append(rew)
      # Calculate how many timesteps we collected this batch   
      t_so_far += np.sum(batch_lens)
      #print("t_so_far ",t_so_far)

      # Calculate V_{phi, k}
      V, _ = self.evaluate(batch_obs, batch_acts)
      #print(" after evaluate ", V , _)
      # ALG STEP 5
      # Calculate advantage
      A_k = batch_rtgs - V.detach()
      # Normalize advantages
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
      for i in range(self.n_updates_per_iteration):
        # Calculate V_phi and pi_theta(a_t | s_t)    
        #print(" update no per iteration ",i)
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
        # Calculate ratios
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean()
        #print("actor_loss ", actor_loss)
        # Calculate gradients and perform backward propagation for actor 
        # network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        critic_loss = nn.MSELoss()(V, batch_rtgs)
        #print("critic_loss ", critic_loss)
        # Calculate gradients and perform backward propagation for critic network    
        self.critic_optim.zero_grad()    
        critic_loss.backward()    
        self.critic_optim.step()
    return episode_reward
    
  def evaluate(self, batch_obs,batch_acts):
    # Query critic network for a value V for each obs in batch_obs.
    V = self.critic(batch_obs).squeeze()
    #print("in evaluate , value function after critic ", V)
    # Calculate the log probabilities of batch actions using most 
    # recent actor network.
    # This segment of code is similar to that in get_action()
    mean = self.actor(batch_obs)
    #print("after actor ", mean)
    dist = Categorical(mean)
    log_probs = dist.log_prob(batch_acts)
    # Return predicted values V and log probs log_probs
    return V, log_probs
 
  