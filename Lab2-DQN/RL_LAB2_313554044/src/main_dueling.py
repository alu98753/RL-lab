from dueldqn_agent_atari import AtariDQNAgent

import os
import torch
import random
import numpy as np


if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
        "gpu": True,
        "training_steps": 1e6*6,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        "warmup_steps": 20000,
        # "warmup_steps": 2000, # test
        "eps_decay": 1000000,
        "eval_epsilon": 0.01,
        "replay_buffer_capacity": 100000,
        "logdir": 'record/log/a/',
        "update_freq": 4,
        "update_target_freq": 10000,
        "learning_rate": 0.0000625,
        "eval_interval": 100,
        # "eval_interval": 2,    # test
        "eval_episode": 5,
        "env_id": 'ALE/MsPacman-v5',
        "seed": 42  # 新增 seed
    }
    # 設定固定的隨機種子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    agent = AtariDQNAgent(config)
    
    # 檢查是否有模型 checkpoint 需要載入
    checkpoint_path = os.path.join(r"E:\NYCU-Project\Class\RL\RL-lab\Lab2-DQN\RL_LAB2_313554044\src\record\log\a\model_latest_best.pth")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print("Model checkpoint loaded successfully.")
    else:
        print("No model checkpoint found at path:", checkpoint_path)
        exit(1)

        
    agent.evaluate()

    # agent.train()
    
    