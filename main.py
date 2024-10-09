import os
import time
from datetime import datetime
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import logging

from PPO import PPO

def eval_model(epoch):

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0)

    ppo_agent.load(f"./models/ppo_{epoch}.pkl")

    test_episodes = 1
    rewards = []

    for _ in range(test_episodes):
        state = env.reset()[0]
        current_ep_reward = 0
        done = False

        while not done:
            action = ppo_agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            state = next_state

            current_ep_reward += reward

        rewards.append(current_ep_reward)

    return np.mean(rewards)

def train():
    env_name = 'CartPole-v1'

    max_ep_len = 500
    max_training_timesteps = 5e5

    eval_freq = max_ep_len * 4

    K_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    actor_lr = 0.0003
    critic_lr = 0.001
    EPOCH = 10000

    random_seed = 0

    logging.basicConfig(level=logging.INFO, 
                        filename='train.log',
                        filemode='w')

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # env.seed(random_seed)

    ppo_agent = PPO(state_dim, action_dim, actor_lr, critic_lr, gamma, K_epochs, eps_clip)

    time_stamp = 0
    epoch = 0

    while time_stamp <= max_training_timesteps:

        state = env.reset()[0]
        current_ep_reward = 0
        done = False

        epoch += 1
        
        for t in range(1, max_ep_len + 1):

            time_stamp += 1

            action = ppo_agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            state = next_state

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            current_ep_reward += reward

            if time_stamp % eval_freq == 0:

                ppo_agent.update()
                ppo_agent.save(f"./models/ppo_{epoch}.pkl")
                reward_mean = eval_model(epoch)
                # print(f'Epoch {epoch}, train reward: {current_ep_reward}, test reward: {reward_mean}')s
                logging.info(str(epoch) + '\t' + str(current_ep_reward) + '\t' + str(reward_mean))

            if done:
                break

if __name__ == '__main__':
    train()