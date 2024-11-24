import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CarRacingEnvironment:
	def __init__(self, N_frame=4, test=False):
		self.test = test
		if self.test:
			self.env = gym.make('CarRacing-v2', render_mode="rgb_array")
		else:
			self.env = gym.make('CarRacing-v2')
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
	
	def check_car_position(self, obs):
		# cut the image to get the part where the car is
		part_image = obs[60:84, 40:60, :]

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
		grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
		grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
		road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
		grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
		# count the number of pixels in the road and grass
		road_pixel_count = cv2.countNonZero(road_mask)
		grass_pixel_count = cv2.countNonZero(grass_mask)

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, part_image)

		return road_pixel_count, grass_pixel_count

	def calculate_drift_angle(self, obs):
		steering = self.env.car.hull.angularVelocity
		drift_angle = abs(steering) * 30  # 簡化計算
		return drift_angle

	def update_visited_tiles(self, info):
		if 'tile_visited_count' in info:
			current_tile_count = info['tile_visited_count']
			if current_tile_count > self.prev_tile_visited_count:
				for i in range(self.prev_tile_visited_count, current_tile_count):
					self.visited_tiles.add(i)
				self.prev_tile_visited_count = current_tile_count
    
	def is_road_straight(self, obs):
		road_ahead = obs[0:40, :, :]  

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)

		road_mask = cv2.inRange(road_ahead, road_color_lower, road_color_upper)

		center_positions = []
		rows = [5, 10, 15, 20, 25, 30, 35]  
		for row in rows:
			road_indices = np.where(road_mask[row] > 0)[0]
			if len(road_indices) > 0:
				center_x = np.mean(road_indices)
				center_positions.append(center_x)
		if len(center_positions) >= 2:
			std_dev = np.std(center_positions)
			if std_dev < 5:
				return True  
			else:
				return False 
		else:
			return False

	def is_road_turning(self, obs):
		road_ahead = obs[0:40, :, :]  # Take the top 40 rows

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)

		road_mask = cv2.inRange(road_ahead, road_color_lower, road_color_upper)

		center_positions = []
		valid_rows = []
		rows = [5, 10, 15, 20, 25, 30, 35]  
		for row in rows:
			road_indices = np.where(road_mask[row] > 0)[0]
			if len(road_indices) > 0:
				center_x = np.mean(road_indices)
				center_positions.append(center_x)
				valid_rows.append(row)       

		if len(center_positions) >= 2:
			z = np.polyfit(valid_rows, center_positions, 1)
			slope = z[0]
			if abs(slope) > 0.5:
				if slope > 0:
					return 'left'  
				else:
					return 'right'  
			else:
				return None  
		else:
			return None  


	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1
  
     	# 獲取車輛位置資訊
		road_pixel_count, grass_pixel_count = self.check_car_position(obs)
		info["road_pixel_count"] = road_pixel_count
		info["grass_pixel_count"] = grass_pixel_count
    	
		speed = np.linalg.norm(self.env.car.hull.linearVelocity)
		info['speed'] = speed  # 獲取車速資訊


		# my reward shaping strategy, you can try your own
		if road_pixel_count < 10:
			# terminates = True
			reward = -50
		else:
			if self.ep_len > 40:
				img_size = 20 * 24
				car_size = img_size - road_pixel_count - grass_pixel_count
				road_ratio = road_pixel_count / (img_size - car_size)
				reward += 5 * min(0.01, road_ratio - 0.5)

				if road_ratio < 0.15:
					reward -=  0.001 # 偏離賽道的懲罰

				self.update_visited_tiles(info)
    
				if self.is_road_straight(obs):
					if speed > 0.8:
						reward += 0.2  # 鼓勵高速行駛
					if abs(action[0]) < 0.1:
						reward += 0.02
					if action[2] > 0.5:
						reward -= 0.05  # 懲罰頻繁剎車

				# 獎勵訪問新的賽道圖塊
				if 'tile_visited_count' in info:
					# if road_pixel_count < 10:
					# 	terminates = True
					if info['tile_visited_count'] > len(self.visited_tiles):
						new_tiles = info['tile_visited_count'] - len(self.visited_tiles)
						reward += new_tiles * (1000 / self.total_tiles) 


				# 新增甩尾機制
				turn_direction = self.is_road_turning(obs)
				if turn_direction in ['left', 'right']:
					steering = action[0]
					drift_triggered = abs(steering) > 0.5 and speed > 0.6
					drift_angle = self.calculate_drift_angle(obs)

					if drift_triggered and 10 <= drift_angle <= 30:
						reward += 2  
					elif drift_triggered and drift_angle > 30:
						reward -= 2  
				else:
					if abs(action[0]) > 0.5 and speed > 0.6:
						reward -= 2  

				if info.get('lap_complete'):
					reward += 50  
					terminates = True 

		# convert to grayscale
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	# def reset(self):
	# 	obs, info = self.env.reset()
	# 	self.ep_len = 0
	# 	obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

	# 	# frame stacking
	# 	for _ in range(self.frames.maxlen):
	# 		self.frames.append(obs)
	# 	obs = np.stack(self.frames, axis=0)

	# 	return obs, info

	def reset(self, seed=None):
		obs, info = self.env.reset(seed=seed)
		self.ep_len = 0
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)  # 96x96

		self.total_tiles = info.get('track_tile_num', 1000)
		self.visited_tiles = set()
		self.prev_tile_visited_count = 0

		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
 
	# 錄影要回傳render
	def render(self):
		self.env.render()
	
		return self.env.render()

	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(test=True)
	obs, info = env.reset()
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		action[2] = 0.0
		obs, reward, terminates, truncates, info = env.step(action)
		print(f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
