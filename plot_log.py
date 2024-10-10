import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

reard_list = []
file = './train.log'

with open(file, 'r') as f:
    for line in f:
        content = line.split('\t')
        reard_list.append(float(content[-1]))

window_size = 20

smoothed_rewards = moving_average(reard_list, window_size)

plt.figure(figsize=(10, 5))

plt.plot(reard_list, label='Original Reward', color='lightcoral', alpha=0.7)

plt.plot(smoothed_rewards, label='Smoothed Reward', color='orange')

plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Training Reward')
plt.legend()

plt.savefig('train_reward.png')
