import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
from gym.wrappers import RecordVideo
from moviepy.editor import ImageSequenceClip
import os
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
        
class RND_PPO(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.num_envs = config["num_envs"]  # 設置 self.num_envs


		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": self.num_envs,
			})

		self.writer = SummaryWriter(config["logdir"])
    
	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError

	def train(self):
		# 初始化環境
		observations, infos = self.env.reset()
		episode_rewards = np.zeros(self.num_envs)
		episode_lens = np.zeros(self.num_envs)
		start_time = time.time()
		save_count=0
		# 初始化运行平均和标准差

		while self.total_time_step <= self.training_steps:
			# decide actions & do action
			actions,logp_pis, values  = self.decide_agent_actions(observations)
			next_observations, rewards, terminates, truncates, infos = self.env.step(actions)
   
			# 计算内在奖励并存储
			intrinsic_rewards = self.compute_intrinsic_reward(observations)
			total_rewards  = rewards + self.intrinsic_coefficient * intrinsic_rewards

   			# 將轉移存入 replay buffer
			for i in range(self.num_envs):
				obs = {}
				obs["observation_2d"] = np.asarray(observations[i], dtype=np.float32)
				self.gae_replay_buffer.append(i, {
					"observation": obs,       # shape = (4,84,84)
					"action": actions[i],     # shape = (1,)
					"reward": total_rewards[i],       # shape = ()
					"extrinsic_reward": rewards[i],   # 外在奖励
					"intrinsic_reward": intrinsic_rewards[i],  # 内在奖励
     				"value": values[i],       # shape = ()
					"logp_pi": logp_pis[i],   # shape = ()
					"done": terminates[i],    # shape = ()
				})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()
     
			# 累計獎勵和回合長度
			episode_rewards += rewards
			episode_lens += 1
   
			# 如果該回合結束（terminate 或 truncate 為真），會使用 TensorBoard 記錄該回合的總獎勵與長度
			for i in range(self.num_envs):
				if terminates[i] or truncates[i]:
					# 添加到獎勵歷史
					if i == self.num_envs - 1:
						self.writer.add_scalar('Train/Episode Reward', episode_rewards[i], self.total_time_step)
						self.writer.add_scalar('Train/Episode Len', episode_lens[i], self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}] episode reward: {episode_rewards[i]}  episode len: {episode_lens[i]}")
					episode_rewards[i] = 0
					episode_lens[i] = 0
					self.writer.add_scalar('Train/Intrinsic Reward', intrinsic_rewards.mean(), self.total_time_step)
					self.writer.add_scalar('Train/Extrinsic Reward', rewards.mean(), self.total_time_step)

     
			# 7 環境更新、訓練步驟計數
			observations = next_observations
			self.total_time_step += self.num_envs

			# 8 在設定的間隔回合數內，進行評估，並保存當前模型權重
			if self.total_time_step % self.eval_interval == 0:
				print("execute time:", round( float(time.time()-start_time)/3600,2) ,"hr" )
				avg_score = self.evaluate()
				# save model checkpoint
				latest_save_path = os.path.join(self.writer.log_dir, "model_latest.pth")
				self.save(latest_save_path)
				print(f"Model saved to {latest_save_path}")

				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				save_count += 1
				if(save_count % 50 )==0:
					self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
					save_count =1

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
  
		# 設置影片保存路徑
		# video_folder  = os.path.join(self.writer.log_dir, "./Lab3-PPO/Code/evaluation_videos" + time.strftime("%Y%m%d-%H%M%S"))
		# if not os.path.exists(video_folder ):
		# 	os.makedirs(video_folder )
   
		# 包裝環境以錄製視頻
		frames = []  # 用於存儲所有幀
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# 渲染當前幀並將其添加到幀列表中
				# frame = self.test_env.render()
				# if frame is not None:
				# 	frames.append(frame)
     
				action, _, _, = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(action[0])
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
  
		# # 使用 MoviePy 生成視頻
		# if frames:
		# 	video_path = os.path.join(os.getcwd(),video_folder, "evaluation_output.mp4")
		# 	clip = ImageSequenceClip(frames, fps=30)  # FPS 可以根據您的需要進行調整
		# 	clip.write_videofile(video_path, codec='libx264')
		return avg
		
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()