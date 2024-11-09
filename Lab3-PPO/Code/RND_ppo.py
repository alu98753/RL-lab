import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from models.atari_model import AtariNet
import gym
from gym.vector import SyncVectorEnv

from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
import warnings
os.environ["SDL_VIDEODRIVER"] = "dummy"

from RND_base_agent import RND_PPO
from models.RNDModel import RNDModel
from RND_base_agent import RunningMeanStd
from gym.wrappers import TimeLimit

# 忽略特定的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")
class PPOWithRND(RND_PPO):
	def __init__(self, config):
		super(PPOWithRND, self).__init__(config)
  

		# 初始化RND模型和优化器
		input_shape = (4, 84, 84)  # 根据您的环境调整
		output_size = 512  # 可以根据需要调整
		self.rnd_model = RNDModel(input_shape, output_size).to(self.device)
		self.rnd_learning_rate = config["rnd_learning_rate"]
		self.rnd_optimizer = torch.optim.Adam(self.rnd_model.predictor.parameters(), lr=self.rnd_learning_rate)
		self.intrinsic_coefficient = config["intrinsic_coefficient"]
		self.rnd_update_epochs = config["rnd_update_epochs"]
		self.rnd_rms = RunningMeanStd(shape=())

		### TODO ###
		# initialize env
		def make_env(render_mode="rgb_array", noop_max=30,max_episode_steps=10000):
			env = gym.make(f'ALE/{config["env_id"]}', render_mode=render_mode, frameskip=1)
			env = atari_preprocessing.AtariPreprocessing(env, frame_skip=4, noop_max=noop_max)
			env = FrameStack(env, 4)
			env = TimeLimit(env, max_episode_steps=max_episode_steps)
			return env

		self.env = gym.vector.AsyncVectorEnv([
			lambda: make_env() for _ in range(self.num_envs)
		])


		### TODO ###
		# initialize test_env
		self.test_env = make_env("rgb_array", noop_max=0)

		initial_observation, _ = self.env.reset()
		print(f"Initial observation shape: {initial_observation.shape}")
		test_initial_observation, _ = self.test_env.reset()
		print(f"Test environment initial observation shape: {test_initial_observation.shape}")
        # 確保觀察值形狀符合預期
		# Initial observation shape: (256, 4, 84, 84)
		# Test environment initial observation shape: ( 4, 84, 84)

		self.net = AtariNet(self.env.single_action_space.n)
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr , eps=1e-5)
  
  
  
	def compute_intrinsic_reward(self, observations):
		obs_tensor = torch.from_numpy(observations).float().to(self.device)
		predictor_output, target_output = self.rnd_model(obs_tensor)
		intrinsic_rewards = (predictor_output - target_output).pow(2).mean(dim=1).cpu().detach().numpy()

		# 更新运行平均和标准差
		self.rnd_rms.update(intrinsic_rewards)

		# 对内在奖励进行归一化
		normalized_intrinsic_rewards = (intrinsic_rewards - self.rnd_rms.mean) / (np.sqrt(self.rnd_rms.var) + 1e-8)

		return normalized_intrinsic_rewards



 
	def update_rnd(self, observations):
		batch_size = 64  # 根据您的显存大小调整 batch_size
		num_samples = observations.shape[0]
		rnd_loss_total = 0.0

		# 将数据打乱
		indices = np.random.permutation(num_samples)
		observations = observations[indices]

		# 分批处理
		for start_idx in range(0, num_samples, batch_size):
			end_idx = min(start_idx + batch_size, num_samples)
			batch_obs = observations[start_idx:end_idx]
			obs_tensor = torch.from_numpy(batch_obs).float().pin_memory().to(self.device, non_blocking=True)
			predictor_output, target_output = self.rnd_model(obs_tensor)
			rnd_loss = (predictor_output - target_output).pow(2).mean()
			self.rnd_optimizer.zero_grad()
			rnd_loss.backward()
			self.rnd_optimizer.step()
			rnd_loss_total += rnd_loss.item() * (end_idx - start_idx)

		# 计算平均 RND 损失
		avg_rnd_loss = rnd_loss_total / num_samples

		# 将平均 RND 损失记录到 TensorBoard
		self.writer.add_scalar('RND/Loss', avg_rnd_loss, self.total_time_step)



  
	### TODO ###
	def decide_agent_actions(self, observation, eval=False):
		# 將觀察值轉換為 PyTorch 張量，並添加批次維度
		# 假設觀察值為 numpy array，形狀為 [num_envs, channels, height, width]
		observation = observation.__array__()
		# print(observation)
		if len(observation.shape) == 3:
			observation = np.expand_dims(observation, axis=0)
	
		obs_tensor = torch.from_numpy(observation).to(self.device, dtype=torch.float32)
		
		if eval:
			with torch.no_grad():
				# 獲取策略（pi）、價值（v）、動作（action）和對數概率（logp）
				action, log_prob, value, entropy = self.net(obs_tensor, eval=True)
		else:
			action, log_prob, value, entropy = self.net(obs_tensor)
		
		# 將動作、價值和對數概率轉換為 numpy，並移到 CPU 上
		action = action.cpu().detach().numpy()
		value = value.cpu().detach().numpy()
		log_prob = log_prob.cpu().detach().numpy()
  
		return action, log_prob, value

	def get_observations_from_buffer(self):
		# 从 Replay Buffer 中提取观测数据
		observations = []
		for i in range(self.num_envs):
			observations.extend(self.gae_replay_buffer.buffer[i]["observation"])
		return observations

	
	def update(self):
		# 初始化損失累積變量

		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

        # 提取批次數據
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
  
		observations = batches["observation"]["observation_2d"]
		# 将 observations 转换为 numpy 数组
		observations = np.array(observations)
  
		# 提取内在奖励
		intrinsic_rewards = batches["intrinsic_reward"]
		mean_intrinsic_reward = np.mean(intrinsic_rewards)
		std_intrinsic_reward = np.std(intrinsic_rewards)
		max_intrinsic_reward = np.max(intrinsic_rewards)
		min_intrinsic_reward = np.min(intrinsic_rewards)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
                # 準備批次數據

				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
				ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
    
				ac_train_batch = torch.from_numpy(ac_train_batch)
				ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
    
				adv_train_batch = torch.from_numpy(adv_train_batch)
				adv_train_batch = (adv_train_batch - adv_train_batch.mean()) / (adv_train_batch.std() + 1e-8)

				adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
    
				logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
				logp_pi_train_batch = logp_pi_train_batch.to(self.device, dtype=torch.float32)
    
				return_train_batch = torch.from_numpy(return_train_batch)
				return_train_batch = return_train_batch.to(self.device, dtype=torch.float32)
				
				# print(f"ob_train_batch shape: {ob_train_batch.shape}")

				# 如果有多餘的維度，將其移除
				if ob_train_batch.dim() == 5 and ob_train_batch.shape[1] == 1:
					ob_train_batch = ob_train_batch.squeeze(1)
				### TODO ###
				# calculate loss and update network
				log_prob, value, entropy = self.net.get_training_data(ob_train_batch, ac_train_batch)
				
				# calculate policy loss
				ratio = torch.exp(log_prob - logp_pi_train_batch) # ratio of new and old probabilities
				surrogate_1 = -ratio * adv_train_batch
				surrogate_2 = -torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_train_batch
				surrogate_loss = torch.max(surrogate_1, surrogate_2).mean()
				# print(f"Surrogate loss shape: {surrogate_loss.shape}")  # 應該是 torch.Size([])

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)

				# 計算熵的平均值 (scalar)
				entropy = entropy.mean()
				# print(f"Entropy shape: {entropy.shape}")  # 應該是 torch.Size([])
              
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()
    
				# 累積損失
				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

  
  		# 記錄損失到 TensorBoard


		        # 更新RND预测网络
        
        
		for _ in range(self.rnd_update_epochs):
			self.update_rnd(observations)

   
		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
      # 打印或记录内在奖励的统计信息
		print(f"Intrinsic Rewards - Mean: {mean_intrinsic_reward}, Std: {std_intrinsic_reward}, Max: {max_intrinsic_reward}, Min: {min_intrinsic_reward}")

		print("Loss: {:}\tSurrogate Loss: {:}\tValue Loss: {:}\tEntropy: {:}".format(
			total_loss / loss_counter,
			total_surrogate_loss / loss_counter,
			total_v_loss / loss_counter,
			total_entropy / loss_counter
			))
		# 清空 replay buffer
		self.gae_replay_buffer.clear_buffer()
