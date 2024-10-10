import gymnasium as gym

env = gym.make("BipedalWalker-v3", hardcore=True)
obs = env.reset()[0]
print(obs)

action_dim = env.action_space.shape[0]
print(action_dim)