from td3_agent_CarRacing import CarRacingTD3Agent
import os
import torch
import random
import numpy as np

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e9,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
        "logdir": '/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/TD3-select',
		"update_freq": 2,
		"eval_interval": 10,
		"eval_episode": 10,
		"seed": 6

	}

	agent = CarRacingTD3Agent(config)
	# agent.train()

	agent.load_and_evaluate(
		"/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/TD3-select/model_14152661_549.pth")
