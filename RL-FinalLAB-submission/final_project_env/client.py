import argparse
import json
import numpy as np
import requests
from ppo import GrayScaleObservation, ResizeObservation
import cv2
import gymnasium as gym


def connect(agent, url: str = 'http://localhost:5000'):
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    args = parser.parse_args()


    class Agent:
        def __init__(self, action_space):
            self.action_space = action_space
            self.frame_stack = None
            self.frame_size = 8
            from stable_baselines3 import PPO
            # python client.py --url http://127.0.0.1:33333
            
            # 7
            self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-FinalLAB/Project/final_project_env/log/PPO_circle_32_7/best_model_24576.zip')   
      

            # 3. #    self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_3/final_model.zip')
            # 5. #    self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_5/best_model_47104.zip')
            # 7. better 3.312  313344 * 0.786432/10   self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_7/best_model_24576.zip')
            # 7.2 best   3.3512 313344* 6.291456/10    self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_7/best_model_196608.zip')            

            # 8.
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_138240.zip')
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_48128.zip')
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_14336.zip')
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_137216.zip')
            
            # 8.2
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_217088.zip')
      
            # 10. best
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_20480.zip')
            # 10.
            # self.model = PPO.load('/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32_8/best_model_41984.zip')
            


        def preprocess_observation(self, obs):
            obs = obs.transpose(1, 2, 0)
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            if self.frame_stack is None:
                self.frame_stack = np.stack([obs] * self.frame_size, axis=2)
            else:
                self.frame_stack = np.roll(self.frame_stack, -1, axis=2)
                self.frame_stack[:, :, -1] = obs
            return self.frame_stack
        def act(self, observation):
            observation = self.preprocess_observation(observation)
            action, _states = self.model.predict(observation, deterministic=True)
            return action


    # Initialize the RL Agent
    import gymnasium as gym

    agent = Agent(
        action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    connect(agent, url=args.url)
# python client.py --url https://competition3.cgi.lab.nycu.edu.tw/12339

# python client.py --url http://127.0.0.1:33333

# 拿 log number * best model的相對位置可得最好model的log number      ex: (624640/2048) *0.27 *2048