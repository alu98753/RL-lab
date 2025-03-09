from td3_agent_CarRacing_different_reward_v3 import CarRacingTD3Agent

if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
  
  
		"logdir": '/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/CarRacing/diff_reward_v3_2/',
		"update_freq": 2,
		"eval_interval": 10,
		"eval_episode": 10,
  		"seed": 2
    
# 6 793
	}
	agent = CarRacingTD3Agent(config)
	# agent.load("/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/TD3-select/model_1880342_530.pth")
	# agent.train()
	# agent.evaluate()
	agent.load_and_evaluate(
		"/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/CarRacing/diff_reward_v3_2/model_1469700_515.pth")

#014953 1 甩尾 # /mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/CarRacing/diff_reward_v3_2/model_269700_722.pth 

#030215 甩尾令人驚嘆	#	/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/CarRacing/diff_reward_v3_2/model_339700_686.pth

#		#/mnt/md0/chen-wei/zi/RL-lab/Lab4-TD3/Code/log/CarRacing/diff_reward_v3_2/model_799700_803.pth
