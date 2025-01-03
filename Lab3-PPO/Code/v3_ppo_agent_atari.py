import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from v3_base_agent import v3_PPOBaseAgent
from models.atari_model import AtariNet
import gym
from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
import warnings
os.environ["SDL_VIDEODRIVER"] = "dummy"
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 忽略特定的 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")
class v3_AtariPPOAgent(v3_PPOBaseAgent):
	def __init__(self, config):
		super(v3_AtariPPOAgent, self).__init__(config)

		### TODO ###
		# initialize env
		def make_env(render_mode="rgb_array", noop_max=30):
			env = gym.make(f'ALE/{config["env_id"]}', render_mode=render_mode)
			env = atari_preprocessing.AtariPreprocessing(env, frame_skip=1, noop_max=noop_max)
			env = FrameStack(env, 4)
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
		# self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr , eps=1e-5)
  
  
		# 初始化優化器
		self.optim = torch.optim.AdamW(
			self.net.parameters(), 
			lr=config["learning_rate"], 
			betas=(0.9, 0.999)
		)

		# 初始化學習率調度器
		self.scheduler = ReduceLROnPlateau(
			self.optim,
			mode='max',  # 假設您希望最大化獎勵
			factor=0.5,  # 每次調整學習率的倍率
			patience=10,  # 在指標未提升時等待的 epoch 數
			verbose=True
		)

		# 初始化熵係數參數
		self.initial_entropy_coef = config["entropy_coefficient"]
		self.final_entropy_coef = config.get("final_entropy_coefficient", 0.0)
		self.entropy_decay_steps = config.get("entropy_decay_steps",  config["entropy_decay_steps"])
		self.current_entropy_coef = self.initial_entropy_coef

		# 初始化熵係數優化器
		self.entropy_coef_target = config.get("entropy_coef_target", 0.01)
		self.entropy_coef_lr = config.get("entropy_coef_lr", 1e-3)
		self.entropy_coef = nn.Parameter(torch.tensor(self.initial_entropy_coef))
		self.optim_entropy = torch.optim.Adam([self.entropy_coef], lr=self.entropy_coef_lr)
  
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
				# print(f"Value loss shape: {v_loss.shape}")  # 應該是 torch.Size([])
				# print(f"Entropy shape: {entropy.shape}")  # 應該是 torch.Size([])

				# 計算熵的平均值 (scalar)
				entropy = entropy.mean()
				# print(f"Entropy shape: {entropy.shape}")  # 應該是 torch.Size([])
              
                # 更新熵係數
				self.current_entropy_coef = self.get_current_entropy_coef()
      
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.current_entropy_coef * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				# 更新學習率調度器（如果使用 ReduceLROnPlateau）
				avg_reward = self.get_average_reward()  # 實現一個方法來計算當前的平均獎勵
				if avg_reward is not None:
					self.scheduler.step(avg_reward)  # 使用位置參數傳遞 avg_reward				# 更新熵係數優化器（如果使用自適應熵係數）
				self.optim_entropy.step()
    
				# 累積損失
				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

  
  		# 記錄損失到 TensorBoard
		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		# print("Loss: {:}\tSurrogate Loss: {:}\tValue Loss: {:}\tEntropy: {:}".format(
		# 	total_loss / loss_counter,
		# 	total_surrogate_loss / loss_counter,
		# 	total_v_loss / loss_counter,
		# 	total_entropy / loss_counter
		# 	))
		        # 清空 replay buffer
		self.gae_replay_buffer.clear_buffer()
