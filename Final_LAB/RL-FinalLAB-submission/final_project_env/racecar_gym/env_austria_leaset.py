from collections import OrderedDict
import gymnasium as gym
import numpy as np
from numpy import array, float32
# noinspection PyUnresolvedReferences
import racecar_gym.envs.gym_api


class RaceEnv(gym.Env):
    camera_name = 'camera_competition'
    motor_name = 'motor_competition'
    steering_name = 'steering_competition'
    """The environment wrapper for RaceCarGym.
    
    - scenario: str, the name of the scenario.
        'austria_competition' or
        'plechaty_competition'
    
    Notes
    -----
    - Assume there are only two actions: motor and steering.
    - Assume the observation is the camera value.
    """
    def __init__(self,
                 scenario: str,
                 render_mode: str = 'rgb_array_birds_eye',
                 reset_when_collision: bool = True,
                 **kwargs):
        self.scenario = scenario.upper()[0] + scenario.lower()[1:]
        self.env_id = f'SingleAgent{self.scenario}-v0'
        self.env = gym.make(id=self.env_id,
                            render_mode=render_mode,
                            reset_when_collision=reset_when_collision,
                            **kwargs)
        self.render_mode = render_mode
        # Assume actions only include: motor and steering
        self.action_space = gym.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=float32)
        # Assume observation is the camera value
        # noinspection PyUnresolvedReferences
        observation_spaces = {k: v for k, v in self.env.observation_space.items()}
        assert self.camera_name in observation_spaces, f'One of the sensors must be {self.camera_name}. Check the scenario file.'
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
        #
        self.cur_step = 0
        self.previous_info = dict()

    def observation_postprocess(self, obs):
        obs = obs[self.camera_name].astype(np.uint8).transpose(2, 0, 1)
        return obs

    def reset(self, *args, **kwargs: dict):
        if kwargs.get("options"):
            kwargs["options"]["mode"] = 'random'
        else:
            kwargs["options"] = {"mode": 'random'}
        self.cur_step = 0
        obs, *others = self.env.reset(*args, **kwargs)
        # print("Initial Observation:", obs)

        obs = self.observation_postprocess(obs)
        self.previous_info['motor'] = 0
        self.previous_info['steering'] = 0
        self.previous_info['state'] = others[0].copy()
        return obs, others

    def step(self, actions):
        self.cur_step += 1
        motor_action, steering_action = actions

        # 添加noise
        motor_scale = 0.005 # dont too big
        steering_scale = 0.01
        motor_action = np.clip(motor_action + np.random.normal(scale=motor_scale), -1., 1.)
        steering_action = np.clip(steering_action + np.random.normal(scale=steering_scale), -1., 1.)

        dict_actions = OrderedDict([
            (self.motor_name, array(motor_action, dtype=float32)),
            (self.steering_name, array(steering_action, dtype=float32))
        ])
        obs, *others = self.env.step(dict_actions)
        obs = self.observation_postprocess(obs)
        reward, done, truncated, state = others
        
        
        reward = 0
        truncated |= (state['time'] >= 100)

        # print(state['velocity'][0])
        # print( state['dist_goal'])
        
        # print(state['pose'][0])
        # print(state['pose'][1])
        # print(state['pose'][2])
        
        
        # 速度類型 reward
        reward += 0.5 * abs(state['velocity'][0])  # 獎勵更高速度
        reward += 0.5 * motor_action
        
            
        # 進度類型reward    

        # checkpoint獎勵
        if state['checkpoint'] > self.previous_info['state']['checkpoint']:
            reward += 20

        # progress 表示車輛在當前賽道上的進度百分比  
        if state['progress'] > self.previous_info['state']['progress']: 
            reward += 500 * (state['progress'] - self.previous_info['state']['progress']) 
        elif state['progress'] == self.previous_info['state']['progress']:   # not moving
            reward -= 0.1

        # 碰撞懲罰
        if state['wall_collision'] or state['n_collision'] > 0:
            reward -= 200  
            done = True 

        # 更新 previous_info
        self.previous_info['motor'] = motor_action.copy()
        self.previous_info['steering'] = steering_action.copy()
        self.previous_info['state'] = state.copy()

        return obs, reward, done, truncated, state




    def render(self):
        return self.env.render()
    
    def force_render(self, render_mode: str = 'rgb_array_follow', **kwargs):
        # 直接呼叫 
        if hasattr(self.env.unwrapped, 'force_render'):
            return self.env.unwrapped.force_render(render_mode=render_mode, **kwargs)
