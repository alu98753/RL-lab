import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

class DQNBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.epsilon = 1.0
		self.eps_min = config["eps_min"]
		self.eps_decay = config["eps_decay"]
		self.eval_epsilon = config["eval_epsilon"]
		self.warmup_steps = config["warmup_steps"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.update_freq = config["update_freq"]
		self.update_target_freq = config["update_target_freq"]
	
        # 創建日誌目錄
		logdir = config["logdir"]
		# 檢查路徑是否已存在且為目錄，若不是則強制創建
		if not os.path.exists(logdir):
			os.makedirs(logdir)
		elif not os.path.isdir(logdir):
			raise NotADirectoryError(f"The logdir '{logdir}' exists but is not a directory. Please remove it or provide a valid directory path.")
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]))
		self.writer = SummaryWriter(logdir)

	@abstractmethod
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		action = None
		return action
	
	def update(self):
		if self.total_time_step % self.update_freq == 0:
			self.update_behavior_network()
		if self.total_time_step % self.update_target_freq == 0:
			self.update_target_network()

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		

	def update_target_network(self):
		self.target_net.load_state_dict(self.behavior_net.state_dict())
	
	def epsilon_decay(self):
		self.epsilon -= (1 - self.eps_min) / self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_min)

	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while True:
				# decide action
				if self.total_time_step < self.warmup_steps:
					action = self.decide_agent_actions(observation, 1.0, self.env.action_space)
				else:
					action = self.decide_agent_actions(observation, self.epsilon, self.env.action_space)
					self.epsilon_decay()

				next_observation, reward, terminate, truncate, info = self.env.step(action)
				self.replay_buffer.append(observation, [action], [reward], next_observation, [int(terminate)])

				#update parameter
				if self.total_time_step >= self.warmup_steps:
					self.update()

				episode_reward += reward
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}  epsilon: {self.epsilon}")
					break
					
				observation = next_observation
				self.total_time_step += 1
				
			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				# 儲存最新的 checkpoint 覆蓋掉上次的
				latest_save_path = os.path.join(self.writer.log_dir, "model_latest.pth")
				self.save(latest_save_path)
				print(f"Model saved to {latest_save_path}")
				# 當次 checkpoint 取特定名稱
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))

				# Tensorborad 存檔
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
  
  
		# 設置視頻保存路徑
		video_dir = os.path.join(self.writer.log_dir, "evaluation_videos" + time.strftime("%Y%m%d-%H%M%S"))
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)

		# 包裝環境以錄製視頻
		self.test_env = RecordVideo(self.test_env, video_dir, disable_logger=True)
  
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				self.test_env.render()
				action = self.decide_agent_actions(observation, self.eval_epsilon, self.test_env.action_space)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action)
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		self.test_env.close()  # 關閉環境，結束錄製

		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.behavior_net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.behavior_net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()




