# Q6: Implement Q-Learning for FrozenLake
import gymnasium as gym
import numpy as np
from common import train_q_learning
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table, rewards = train_q_learning(env, episodes=3000)
print("Learned Q-table:\n", np.round(q_table, 3))
print("Last 100 episode reward average:", np.mean(rewards[-100:]))
env.close()
