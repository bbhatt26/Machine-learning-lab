# Q9: Display the learned Q-table
import gymnasium as gym
import numpy as np
from common import train_q_learning
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table, rewards = train_q_learning(env, episodes=5000)
np.set_printoptions(precision=3, suppress=True)
print("Q-table (rows=states, columns=Left/Down/Right/Up):")
print(q_table)
env.close()
