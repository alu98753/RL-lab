from racecar_gym.env_austria_2 import RaceEnv
# from racecar_gym.env import RaceEnv
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.spaces import Box
import gc
import logging
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
import os
import numpy as np
from utils import *
from collections import OrderedDict
# noinspection PyUnresolvedReferences
from gymnasium.error import DependencyNotInstalled
from stable_baselines3.common.callbacks import CallbackList
# scenario = 'circle_cw_competition_collisionStop'
scenario = 'austria_competition_collisionStop'
reset_when_collision = True if 'austria' in scenario else False
import os

class GrayScaleObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Convert the image observation from RGB to gray scale.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import GrayScaleObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = GrayScaleObservation(gym.make("CarRacing-v2"))
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)
        >>> env = GrayScaleObservation(gym.make("CarRacing-v2"), keep_dim=True)
        >>> env.observation_space
        Box(0, 255, (96, 96, 1), uint8)
    """

    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """Convert the image observation from RGB to gray scale.

        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        gym.utils.RecordConstructorArgs.__init__(self, keep_dim=keep_dim)
        gym.ObservationWrapper.__init__(self, env)

        self.keep_dim = keep_dim

        assert (
            isinstance(self.observation_space, Box)
            and len(self.observation_space.shape) == 3
            and self.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        """Converts the colour observation to greyscale.

        Args:
            observation: Color observations

        Returns:
            Grayscale observations
        """
        import cv2

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation

class ResizeObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Resize the image observation.

    This wrapper works on environments with image observations. More generally,
    the input can either be two-dimensional (AxB, e.g. grayscale images) or
    three-dimensional (AxBxC, e.g. color images). This resizes the observation
    to the shape given by the 2-tuple :attr:`shape`.
    The argument :attr:`shape` may also be an integer, in which case, the
    observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env, shape: tuple[int, int] | int) -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(
            x > 0 for x in shape
        ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

        self.shape = tuple(shape)

        assert isinstance(
            env.observation_space, Box
        ), f"Expected the observation space to be Box, actual type: {type(env.observation_space)}"
        dims = len(env.observation_space.shape)
        assert (
            dims == 2 or dims == 3
        ), f"Expected the observation space to have 2 or 3 dimensions, got: {dims}"

        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation.reshape(self.observation_space.shape)


if __name__ == "__main__":
    def make_env():
        env = RaceEnv(scenario='austria_competition_collisionStop',
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False)
        env = ChangeObs(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env
    CPU = 32 
    env = SubprocVecEnv([lambda: make_env() for i in range(CPU)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)
        
    LOAD_MODEL_DIR = './log/logs_PPO_V2_1/'
    LOG_DIR = './log/logs_PPO_V2/'

    # model_path = os.path.join(LOAD_MODEL_DIR, "final_model")
    # model = PPO.load("/mnt/md0/chen-wei/zi/RL-reference/RL-ref/Project/final_project_env/log/PPO_circle_32/best_model_217088.zip", env=env)  # 載入模型並連結環境
    # model.tensorboard_log=LOG_DIR
    # model.device='cuda'
    # model.use_sde=True
    # 更新學習率
    CHECKPOINT_DIR = './log/PPO_v2_1/'
    # for param_group in model.policy.optimizer.param_groups:
    #     param_group['lr'] = 5e-5
    # model.learning_rate = lambda _: 5e-5

    # model.vf_coef = 0.5 # 降低value loss 
    # model.batch_size = 128        # 修改批量大小
    # model.clip_range =  lambda _: 0.1  # 固定裁剪範圍為 0.2
    # model.ent_coef = 0.001        # 修改熵正則項係數
    
    callback = LoggingCallback(check_freq=2048, save_path=CHECKPOINT_DIR)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda', learning_rate=2e-4, use_sde=True, n_steps=1024, batch_size=64, n_epochs=10, clip_range=0.1, ent_coef=0.005 )

    for param_group in model.policy.optimizer.param_groups:
        print("Learning rate for param group:", param_group['lr'])

    model.learn(total_timesteps=1e7, callback=callback, progress_bar=True)
    model_path = os.path.join(CHECKPOINT_DIR, "final_model")
    model.save(model_path)

