from dqn_agent_atari import AtariDQNAgent
import os

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
        "gpu": True,
        "training_steps": 1e8,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_min": 0.1,
        # "warmup_steps": 20000,
        "warmup_steps": 2000, # test
        "eps_decay": 1000000,
        "eval_epsilon": 0.01,
        "replay_buffer_capacity": 100000,
        "logdir": 'Lab2-DQN/log/DQN_test/',
        "update_freq": 4,
        "update_target_freq": 10000,
        "learning_rate": 0.0000625,
        # "eval_interval": 100,
        "eval_interval": 5,    # test
        "eval_episode": 5,
        "env_id": 'ALE/MsPacman-v5',
        "use_double": False,
        # "obs_type": "rgb",
        # "obs_type": "grayscale", # 設定環境中觀察到的畫面 (observation) 類型
    }
    agent = AtariDQNAgent(config)
    
    # 檢查是否有模型 checkpoint 需要載入
    checkpoint_path = os.path.join(config["logdir"], "model_latest.pth")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print("Model checkpoint loaded successfully.")
            
    agent.train()