import os
import numpy as np
import torch
import argparse
import gymnasium as gym

from PPO import PPO

def main(epoch):

    np.random.seed(epoch)
    torch.manual_seed(epoch)

    env_name = 'CartPole-v1'
    # env = gym.make(env_name, render_mode='human')
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, action_dim, 0, 0, 0, 0, 0)

    eval_time = 3

    ppo_agent.load(f"./models/ppo_{epoch}.pkl")

    reward_list = []
    for _ in range(eval_time):
        done = False
        state = env.reset()[0]
        total_reward = 0
        while not done:
            action = ppo_agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            state = next_state
            total_reward += reward
        reward_list.append(total_reward)
    
    print(f"Average reward: {np.mean(reward_list)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    main(args.epoch)