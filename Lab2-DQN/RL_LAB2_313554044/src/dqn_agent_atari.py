import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# laself.env = ???
		self.env = gym.make(config["env_id"], render_mode="rgb_array") 
		self.env = atari_preprocessing.AtariPreprocessing(self.env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.env = FrameStack(self.env, 4)

		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
	
	# Override below two function from base_agent.py , dot't change base_agent.py # 
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection

		# print("observation.shape before :", observation.shape)
		# observation = np.expand_dims(observation, axis=0)
		# observation = torch.from_numpy(observation)
		# observation = observation.permute(0, 3, 1, 2)  # 改變維度順序，變為 [1, 3, 210, 160]
		# observation = observation.to(self.device, dtype=torch.float32)
		# print("observation.shape after  :", observation.shape)
		if random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			observation = observation.__array__()
			observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)
			q_value = self.behavior_net(observation)
			action = q_value.max(1)[1].item()			
		# print("action:", action)
		return action

		# return NotImplementedError
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		# q_value = ???
		# with torch.no_grad():
			# q_next = ???

			# if episode terminates at next_state, then q_target = reward
			# q_target = ???
        
		# criterion = ???
		# loss = criterion(q_value, q_target)

		# self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		# self.optim.zero_grad()
		# loss.backward()
		# self.optim.step()

		### TODO #### calculate the loss and update the behavior network

		# 1. get Q(s,a) from behavior net
		q_value = self.behavior_net(state).gather(1, action.long())

		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
			# if episode terminates at next_state, then q_target = reward
			q_target = reward + self.gamma * q_next * (1 - done)

		# 4. calculate loss between Q(s,a) and Q_target
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)
		# 5. update behavior net	
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	
	