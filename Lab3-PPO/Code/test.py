import gym

# 列出所有已註冊的環境
for env in gym.envs.registry:
    print(env)