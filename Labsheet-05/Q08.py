# Q8: Train an agent for multiple episodes using Q-Learning
import gymnasium as gym
import numpy as np
from common import train_q_learning
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table, rewards = train_q_learning(env, episodes=5000)
print("Episodes trained:", len(rewards))
print("Success rate in last 100 episodes:", np.mean(rewards[-100:]))
print("Q-table:\n", np.round(q_table, 3))
env.close()
