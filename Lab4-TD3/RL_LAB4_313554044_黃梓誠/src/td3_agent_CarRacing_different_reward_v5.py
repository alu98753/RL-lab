import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv_v5 import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		self.policy_noise = 0.2
		self.noise_clip = 0.5
		# choose Gaussian noise or OU noise

		noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		self.noise = (noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
			if state.ndimension() == 4 and state.shape[-1] in [1, 3]:  # 判斷是否為 channels_last
				state = state.permute(0, 3, 1, 2)  # 將 NHWC -> NCHW
				# 檢查通道數，若不足，補充通道
			# print(state.size(1))
			if state.size(1) == 3:  # 判斷是否為 3 通道
				batch_size, _, height, width = state.size()
				zeros_channel = torch.zeros((batch_size, 1, height, width), device=self.device)  # 增加一個零通道
				state = torch.cat([state, zeros_channel], dim=1)  # 將零通道拼接到輸入中
			action = self.actor_net(state, brake_rate=brake_rate).cpu().numpy().squeeze()
			action += sigma * self.noise.generate()

		return action
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
		# 	# select action a_next from target actor network and add noise for smoothing
			noise = torch.stack(
				((torch.empty_like(action[:, 0]).data.normal_(0, self.policy_noise).clamp(-self.noise_clip, self.noise_clip)),
				(torch.empty_like(action[:, 1]).data.normal_(0, self.policy_noise / 2).clamp(-self.noise_clip / 2, self.noise_clip / 2)),
				(torch.empty_like(action[:, 2]).data.normal_(0, self.policy_noise / 2).clamp(-self.noise_clip / 2, self.noise_clip / 2))),
				dim=1,
			)
			a_next = self.target_actor_net(next_state) + noise
			a_next[:, 0] = a_next[:, 0].clamp(-1, 1)
			a_next[:, 1] = a_next[:, 1].clamp(0, 1)
			a_next[:, 2] = a_next[:, 2].clamp(0, 1)
   
			q_next1 = self.target_critic_net1(next_state, a_next)
			q_next2 = self.target_critic_net2(next_state, a_next)
		# 	# select min q value from q_next1 and q_next2 (double Q learning)
			q_target = reward + self.gamma * torch.min(q_next1, q_next2) * (1 - done)
		
		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		self.critic_net2.zero_grad()
		critic_loss2.backward()
		self.critic_opt2.step()

		## Delayed Actor(Policy) Updates ##
		if self.total_time_step % self.update_freq == 0:
		# 	## update actor ##
		# 	# actor loss
		# 	# select action a from behavior actor network (a is different from sample transition's action)
		# 	# get Q from behavior critic network, mean Q value -> objective function
		# 	# maximize (objective function) = minimize -1 * (objective function)
			action = self.actor_net(state)
			actor_loss = -1 * self.critic_net1(state, action).mean()
		# 	# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
		
